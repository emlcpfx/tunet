/**
 * POST /api/spark/training-jobs
 *
 * Streaming submit pipeline. Responds with text/event-stream and emits
 * one JSON event per phase so the client can show a real progress bar.
 *
 * Event shapes:
 *   {phase: 'validate',  status: 'start'}                 — request received
 *   {phase: 'pack',      status: 'start' | 'done', files?, kb?, ms?}
 *   {phase: 'submit',    status: 'start' | 'done', jobId?, ms?}
 *   {phase: 'upload',    status: 'start' | 'progress' | 'done', sentBytes?, totalBytes?, ms?}
 *   {phase: 'done',      jobId, output: {...}, totalMs}    — terminal success
 *   {phase: 'error',     phase_at, error, jobId?}          — terminal failure
 *
 * Only one phase=done OR phase=error is emitted per request.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import {
  submitJob, uploadInputTarball, GPU_TYPES, DEFAULT_IMAGE, type GpuKey,
  writeDiscoveredFilesBase, getJob, fetchOutputFile, shareSyncBaseUrl,
} from '@/lib/spark'
import { packInputTarball } from '@/lib/spark-packer'
import {
  PRESETS, buildConfig, type PresetKey, type AdvancedOverrides, type JobInputs,
} from '@/lib/spark-presets'
import type { SerializedFormState, TrainingMode, ComputeMode } from '@/lib/spark-form-state'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 60

interface Body {
  name?:    string
  preset?:  string
  gpu?:     string
  /**
   * Training mode. Defaults to 'new' (back-compat with old clients).
   *   - 'new':      train from scratch (or use whatever weights are in the image).
   *   - 'resume':   continue a prior job. Source's `*_tunet_latest.pth` is
   *                 staged under /input/output/<sourceJobName>/ (spark_start.sh
   *                 copies it into /output/<jobName>/) AND the new job's
   *                 output dir is forced to `/output/<sourceJobName>` so
   *                 train.py's auto-resume picks it up. Preset/loss/model
   *                 must match the source — train.py rejects mismatches.
   *   - 'finetune': pick *any* .pth from any prior job; weights are loaded,
   *                 optimizer + step counter start fresh. Permissive — works
   *                 across architectures so long as shape matches.
   */
  mode?: TrainingMode
  source?: {
    jobId:          string
    /** Filename of the .pth inside the source job's output dir. */
    checkpointName: string
    /**
     * If set, the .pth was uploaded from the user's machine via
     * /api/spark/upload-stage with role='checkpoint' rather than fetched
     * from a prior Spark job. `jobId` is the sentinel 'local-upload' in
     * this case and the route reads bytes from <tmp>/tunet-stages/<id>/
     * checkpoint/<checkpointName> instead of calling the Spark API.
     */
    localCheckpointStageId?: string
  }
  inputs?: {
    src_dir?:       string
    dst_dir?:       string
    output_dir?:    string
    val_src_dir?:   string | null
    val_dst_dir?:   string | null
    finetune_from?: string | null
  }
  advanced?: AdvancedOverrides
  idleHoldSeconds?: number
  /**
   * Compute mode. 'instant' (default) = warm-pool guaranteed-availability.
   * 'smart' = preemptible spare capacity, ~60% cheaper but the platform may
   * reclaim the underlying compute mid-run (auto-retries up to the budget).
   */
  computeMode?: ComputeMode
  /**
   * (smart-mode only) How many auto-retries the platform may consume if the
   * underlying compute is preempted. Default 1, range [0, 5]. Ignored on
   * instant-mode jobs.
   */
  maxRetriesOnInterrupt?: number
  /**
   * Optional: a stageId from /api/spark/upload-stage. If present, the staged
   * data folders (src/, dst/, val_src/, val_dst/) are bundled into the
   * tarball under data/<role>/ and the config is rewritten to point at
   * /input/data/<role>/. Overrides any inputs.src_dir / inputs.dst_dir.
   */
  stageId?: string
  /**
   * Optional training-alert prefs. The cron poller (api/cron/training-alerts)
   * reads these from job.env to decide whether and where to send notifications.
   */
  alerts?: {
    email?:     string   // recipient
    plateau?:   boolean  // notify on plateau / "consider stopping"
    diverging?: boolean  // notify when loss is going up
  }
  /**
   * Serialized full form state — stashed verbatim in env.TUNET_FORM_STATE so a
   * future "Clone job" can rehydrate the new-job form. Opaque to the trainer.
   */
  formState?: SerializedFormState
  /**
   * Optional benchmark mode. When >0, spark_start.sh runs train.py with
   * `--benchmark-steps N`: it warms up briefly, times N steps, logs
   * `STEP_RATE: X.Y` and exits. Used to calibrate the cost estimator's
   * baselineStepsPerSec() in spark-presets.ts. Typical: 200.
   */
  benchmarkSteps?:  number
  /** Warmup steps to discard before timing (default 20). */
  benchmarkWarmup?: number
}

export async function POST(req: Request) {
  let body: Body
  try {
    body = await req.json()
  } catch {
    return jsonError('Invalid JSON', 400)
  }

  // ── Validate up front (before opening the stream so errors are normal HTTP) ─
  if (!body.name?.trim()) return jsonError('name is required', 400)
  if (!body.preset || !(body.preset in PRESETS))
    return jsonError(`preset must be one of: ${Object.keys(PRESETS).join(', ')}`, 400)
  if (!body.gpu || !(body.gpu in GPU_TYPES))
    return jsonError(`gpu must be one of: ${Object.keys(GPU_TYPES).join(', ')}`, 400)
  // ── Resolve stage (if provided) and verify dir exists ────────────────────
  // When a stage is provided we override the src/dst paths to point at
  // /input/data/<role>/ — that's where the packer will land them inside the
  // tarball, which the agent extracts to /input/.
  let stageDir: string | null = null
  let stageRoles: Set<string> = new Set()
  if (body.stageId) {
    const candidate = path.join(os.tmpdir(), 'tunet-stages', body.stageId)
    if (!fs.existsSync(candidate)) {
      return jsonError(`stage ${body.stageId} not found (was it uploaded?)`, 400)
    }
    stageDir = candidate
    for (const e of fs.readdirSync(candidate, { withFileTypes: true })) {
      if (e.isDirectory()) stageRoles.add(e.name)
    }
    if (!stageRoles.has('src') || !stageRoles.has('dst')) {
      return jsonError(`stage ${body.stageId} must contain at least src/ and dst/ subfolders`, 400)
    }
  } else if (!body.inputs?.src_dir || !body.inputs?.dst_dir) {
    return jsonError('Either upload data via stageId or provide inputs.src_dir + inputs.dst_dir', 400)
  }

  const name      = body.name.trim()
  const preset    = PRESETS[body.preset as PresetKey]
  const gpuKey    = body.gpu as GpuKey
  const sku       = GPU_TYPES[gpuKey].sku
  const safeName  = name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64) || 'tunet'

  // ── Resolve mode (new / resume / finetune) ─────────────────────────────
  // Resume / finetune both reference a prior job + checkpoint. We fetch the
  // .pth from ShareSync server-side (browser doesn't have the bearer) and
  // hand it to the packer as an extraFile. spark_start.sh handles seeding
  // for resume; for finetune we drop it at /input/finetune.pth and point
  // config.training.finetune_from at it.
  const mode: TrainingMode = body.mode ?? 'new'
  if (mode !== 'new' && mode !== 'resume' && mode !== 'finetune') {
    return jsonError(`mode must be 'new', 'resume', or 'finetune'`, 400)
  }

  let resumeOutputSubdir: string | null = null  // for resume mode, the *source* job's output subdir
  let extraFiles: { relPath: string; data: Buffer }[] | undefined
  let finetunePath: string | null = body.inputs?.finetune_from ?? null

  if (mode === 'resume' || mode === 'finetune') {
    if (!body.source?.jobId)          return jsonError(`mode='${mode}' requires source.jobId`, 400)
    if (!body.source?.checkpointName) return jsonError(`mode='${mode}' requires source.checkpointName`, 400)
    if (body.source.checkpointName.includes('/') || body.source.checkpointName.includes('..')) {
      return jsonError(`source.checkpointName must be a bare filename`, 400)
    }
    if (!body.source.checkpointName.endsWith('.pth')) {
      return jsonError(`source.checkpointName must end in .pth`, 400)
    }

    // Two source paths: local-upload (the user staged a .pth via
    // /api/spark/upload-stage from their machine) or Spark-source
    // (the .pth lives in a prior Spark job's output ShareSync dir).
    // Local-upload skips the Spark Graph lookup entirely.
    const localStageId = typeof body.source.localCheckpointStageId === 'string'
      ? body.source.localCheckpointStageId
      : null
    let ckptBuf: Buffer
    let sourceSubdir: string

    if (localStageId) {
      const ckptDir = path.join(os.tmpdir(), 'tunet-stages', localStageId, 'checkpoint')
      const ckptPath = path.join(ckptDir, body.source.checkpointName)
      if (!fs.existsSync(ckptPath)) {
        return jsonError(
          `local checkpoint ${body.source.checkpointName} not found at ${ckptPath} ` +
            `(stage may have been cleaned up — re-upload and retry)`,
          400,
        )
      }
      try {
        ckptBuf = await fs.promises.readFile(ckptPath)
      } catch (e) {
        return jsonError(`read local checkpoint failed: ${e instanceof Error ? e.message : 'unknown'}`, 500)
      }
      // For local resume there's no prior Spark job to inherit an output
      // subdir from. Use the new job's safeName so the seeded .pth lands at
      // /output/<safeName>/ and train.py's auto-resume picks it up there.
      sourceSubdir = safeName
    } else {
      // Spark-source path (unchanged): look up the prior job + fetch its .pth.
      let sourceJob
      try {
        sourceJob = await getJob(body.source.jobId)
      } catch (e) {
        return jsonError(`failed to load source job: ${e instanceof Error ? e.message : 'unknown'}`, 502)
      }
      if (!sourceJob) return jsonError(`source job ${body.source.jobId} not found`, 404)
      if (!shareSyncBaseUrl(sourceJob)) {
        return jsonError(`source job has no resolvable ShareSync base URL`, 503)
      }
      const sub = (() => {
        const cmd = sourceJob.command
        if (!Array.isArray(cmd) || cmd.length < 3) return null
        const arg = cmd[2]
        if (typeof arg !== 'string' || !arg.startsWith('/output/')) return null
        return arg.slice('/output/'.length).split('/').filter(Boolean)[0] ?? null
      })()
      if (!sub) {
        return jsonError(`source job ${body.source.jobId} has no recoverable output subdir`, 422)
      }
      sourceSubdir = sub

      // Pull the .pth bytes from ShareSync. fetchOutputFile resolves the URL
      // via the same logic the rest of the proxy uses, so any job that the
      // file-proxy can read is fetchable here.
      let upstream: Response
      try {
        upstream = await fetchOutputFile(sourceJob, body.source.checkpointName)
      } catch (e) {
        return jsonError(`fetch checkpoint failed: ${e instanceof Error ? e.message : 'unknown'}`, 502)
      }
      if (!upstream.ok) {
        return jsonError(
          `checkpoint ${body.source.checkpointName} not found in source output (HTTP ${upstream.status})`,
          upstream.status === 404 ? 404 : 502,
        )
      }
      ckptBuf = Buffer.from(await upstream.arrayBuffer())
    }

    if (mode === 'resume') {
      // train.py picks up checkpoints from <output_dir>/<prefix>_tunet_latest.pth.
      // Place ours at /input/output/<sourceSubdir>/<filename> — spark_start.sh's
      // line 53-62 cp's that into /output/<sourceSubdir>/. We then point the
      // new run's output_dir at /output/<sourceSubdir> so trainer + seeded
      // file end up in the same place.
      resumeOutputSubdir = sourceSubdir
      extraFiles = [{
        relPath: `output/${sourceSubdir}/${body.source.checkpointName}`,
        data:    ckptBuf,
      }]
    } else {
      // Fine-tune: drop the .pth at /input/finetune.pth and let buildConfig
      // wire training.finetune_from to it. Optimizer + step counter reset.
      extraFiles = [{ relPath: 'finetune.pth', data: ckptBuf }]
      finetunePath = '/input/finetune.pth'
    }
  }

  // Resume forces output_dir to the source's dir; new/finetune use the new name.
  const outputDir = mode === 'resume' && resumeOutputSubdir
    ? `/output/${resumeOutputSubdir}`
    : (body.inputs?.output_dir ?? `/output/${safeName}`)

  const jobInputs: JobInputs = stageDir ? {
    // When data is staged in the tarball, paths are deterministic
    src_dir:       '/input/data/src',
    dst_dir:       '/input/data/dst',
    output_dir:    outputDir,
    val_src_dir:   stageRoles.has('val_src') ? '/input/data/val_src' : null,
    val_dst_dir:   stageRoles.has('val_dst') ? '/input/data/val_dst' : null,
    finetune_from: finetunePath,
  } : {
    src_dir:       body.inputs!.src_dir!,
    dst_dir:       body.inputs!.dst_dir!,
    output_dir:    outputDir,
    val_src_dir:   body.inputs?.val_src_dir   ?? null,
    val_dst_dir:   body.inputs?.val_dst_dir   ?? null,
    finetune_from: finetunePath,
  }
  const config = buildConfig(preset, jobInputs, body.advanced ?? {})

  // ── Open the streaming response ──────────────────────────────────────────
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      const t0 = Date.now()

      const send = (event: Record<string, unknown>) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }

      // Heartbeat so proxies don't kill the stream during long phases
      const heartbeat = setInterval(() => {
        try { controller.enqueue(encoder.encode(': keepalive\n\n')) } catch { /* closed */ }
      }, 15_000)

      try {
        send({ phase: 'validate', status: 'done' })

        // ── Pack ────────────────────────────────────────────────────────
        send({ phase: 'pack', status: 'start' })
        const tPack = Date.now()
        const pack = await packInputTarball({
          config,
          stageDir:   stageDir ?? undefined,
          extraFiles: extraFiles,
        })
        send({
          phase:  'pack',
          status: 'done',
          files:  pack.fileCount,
          kb:     Math.round(pack.compressedSize / 1024),
          ms:     Date.now() - tPack,
        })

        // ── Submit ──────────────────────────────────────────────────────
        // We submit twice in the rare case we need to stash the
        // shareSyncBaseUrl in env (Spark's job detail endpoint only echoes
        // the path portion). To avoid two submits we stash the *predicted*
        // base from the env var here; if SPARK_FILES_BASE_URL isn't set the
        // detail UI will fall back to the path-only display.
        send({ phase: 'submit', status: 'start' })
        const tSubmit = Date.now()
        const filesBase = process.env.SPARK_FILES_BASE_URL?.replace(/\/+$/, '')
        const computeMode: ComputeMode = body.computeMode === 'smart' ? 'smart' : 'instant'
        const maxRetriesOnInterrupt = computeMode === 'smart'
          ? Math.max(0, Math.min(5, Math.floor(body.maxRetriesOnInterrupt ?? 2)))
          : undefined

        const submitResp = await submitJob({
          name,
          instanceType:    sku,
          image:           DEFAULT_IMAGE,
          command:         ['bash', '/input/spark_start.sh', outputDir, '/input/config.yaml'],
          idleHoldSeconds: body.idleHoldSeconds ?? 0,
          mode:            computeMode,
          ...(maxRetriesOnInterrupt !== undefined ? { maxRetriesOnInterrupt } : {}),
          // Spark doesn't echo the submit `name` back in list/detail responses,
          // so we stash the friendly name + preset/gpu in env. The container
          // ignores these (they're just metadata for our own UI to read back).
          env: {
            TUNET_JOB_NAME:  name,
            TUNET_PRESET:    preset.key,
            TUNET_GPU:       gpuKey,
            // Surface the compute mode in env so the job-detail UI can render
            // a "SmartCompute" badge without re-fetching the full submit body.
            ...(computeMode === 'smart' ? { TUNET_COMPUTE_MODE: 'smart' } : {}),
            // Training-mode (new/resume/finetune) — distinct from compute mode.
            // Read by jobs-list / detail to show a "Resume of" / "Fine-tune of"
            // badge. Trainer ignores these.
            ...(mode !== 'new' ? { TUNET_MODE: mode } : {}),
            ...(body.source?.jobId          ? { TUNET_SOURCE_JOB_ID:    body.source.jobId          } : {}),
            ...(body.source?.checkpointName ? { TUNET_SOURCE_CHECKPOINT: body.source.checkpointName } : {}),
            ...(filesBase ? { TUNET_FILES_BASE: filesBase } : {}),
            // Training alert prefs (read by api/cron/training-alerts).
            // Only include if email is set — empty email = opted out.
            ...(body.alerts?.email?.trim()
              ? {
                  TUNET_ALERT_EMAIL:     body.alerts.email.trim(),
                  ...(body.alerts.plateau   ? { TUNET_ALERT_PLATEAU:   '1' } : {}),
                  ...(body.alerts.diverging ? { TUNET_ALERT_DIVERGING: '1' } : {}),
                }
              : {}),
            // Full form state for "Clone job" rehydration. Trainer ignores this.
            ...(body.formState
              ? { TUNET_FORM_STATE: JSON.stringify(body.formState) }
              : {}),
            // Benchmark mode: spark_start.sh reads BENCHMARK_STEPS and
            // appends `--benchmark-steps N` to the train.py command.
            ...(body.benchmarkSteps && body.benchmarkSteps > 0
              ? {
                  BENCHMARK_STEPS: String(body.benchmarkSteps),
                  ...(body.benchmarkWarmup && body.benchmarkWarmup > 0
                    ? { BENCHMARK_WARMUP: String(body.benchmarkWarmup) }
                    : {}),
                }
              : {}),
          },
        })
        send({
          phase:  'submit',
          status: 'done',
          jobId:  submitResp.jobId,
          ms:     Date.now() - tSubmit,
        })

        // Cache the discovered ShareSync base URL — the submit response is
        // the only authoritative source. Subsequent file proxy requests for
        // jobs that don't have TUNET_FILES_BASE in their env will use this.
        if (submitResp.output?.shareSyncBaseUrl) {
          writeDiscoveredFilesBase(
            submitResp.output.shareSyncBaseUrl,
            submitResp.output.shareSyncPath ?? submitResp.outputShareSyncPath,
          )
        }

        // ── Upload ──────────────────────────────────────────────────────
        send({
          phase:       'upload',
          status:      'start',
          totalBytes:  pack.compressedSize,
        })
        const tUpload = Date.now()
        // We can't easily report progress from a single fetch PUT in Node,
        // but uploads are short (~3-8s for ~800 KB). Emit a synthetic
        // "halfway" update so the bar moves once for visual reassurance.
        const halfwayTimer = setTimeout(() => {
          send({
            phase:      'upload',
            status:     'progress',
            sentBytes:  Math.floor(pack.compressedSize / 2),
            totalBytes: pack.compressedSize,
          })
        }, 1500)

        try {
          await uploadInputTarball(submitResp.input.uploadUrl, pack.buffer)
        } finally {
          clearTimeout(halfwayTimer)
        }
        send({
          phase:      'upload',
          status:     'done',
          sentBytes:  pack.compressedSize,
          totalBytes: pack.compressedSize,
          ms:         Date.now() - tUpload,
        })

        // ── Cleanup stage on success ─────────────────────────────────────
        if (stageDir) {
          try {
            await fs.promises.rm(stageDir, { recursive: true, force: true })
          } catch { /* ignore cleanup failures */ }
        }

        // ── Done ────────────────────────────────────────────────────────
        send({
          phase:   'done',
          jobId:   submitResp.jobId,
          output:  submitResp.output,
          totalMs: Date.now() - t0,
        })
      } catch (e) {
        send({
          phase:    'error',
          error:    e instanceof Error ? e.message : String(e),
          totalMs:  Date.now() - t0,
        })
      } finally {
        clearInterval(heartbeat)
        controller.close()
      }
    },
  })

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type':       'text/event-stream',
      'Cache-Control':      'no-cache, no-transform',
      'Connection':         'keep-alive',
      'X-Accel-Buffering':  'no',
    },
  })
}

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
