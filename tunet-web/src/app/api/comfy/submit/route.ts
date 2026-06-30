/**
 * POST /api/comfy/submit — submit a ComfyUI job to Spark (EZ-Comfy).
 *
 * Body: { presetKey, values, stageId?, inputName?, gpu?, mode?, idleHoldSeconds?,
 *         loras?, readyTimeout? }
 *
 * Streams Server-Sent Events: { phase: 'validate'|'pack'|'submit'|'upload'|'done'|'error', ... }
 * mirroring the training-jobs flow. The input clip (if any) is read from the
 * upload-stage dir (<tmp>/tunet-stages/<stageId>/comfy_input/<inputName>).
 */
import { auth } from '@/auth'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import {
  submitJob, uploadInputTarball, getToken, getRefreshContext, writeDiscoveredFilesBase,
} from '@/lib/spark'
import {
  loadComfyPreset, loadComfyWorkflow, buildComfyPatches, buildComfyEnv,
  comfyInstanceType, comfyCommand, packComfyTarball, comfySecondaryParam,
  buildComfyBatchManifest, detectSequenceDir,
  type ComfyLora, type ComfyPatch, type ComfyBatchItem,
} from '@/lib/comfy'
import { buildComfyFormState } from '@/lib/comfy-form-state'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 300

interface Body {
  presetKey:       string
  values?:         Record<string, unknown>
  stageId?:        string
  inputName?:      string
  secondaryName?:  string          // two-input presets: face image / mask (comfy_input2)
  batch?:          boolean         // BATCH: render every staged input in ONE job
  gpu?:            string
  mode?:           'instant' | 'smart'
  idleHoldSeconds?: number
  loras?:          ComfyLora[]
  readyTimeout?:   number
}

interface BatchAssembly {
  inputs:    { path: string; basename: string }[]
  seqGroups: { subdir: string; files: { path: string; basename: string }[] }[]
  items:     ComfyBatchItem[]
}

/** Discover a batch's inputs from the upload stage: videos under comfy_input/,
 * image-sequence folders under comfy_seq/<tag>/. Returns null on no inputs. */
function assembleBatch(stageId: string): BatchAssembly | { error: string } {
  const root = path.join(os.tmpdir(), 'tunet-stages', stageId)
  const inputs: BatchAssembly['inputs'] = []
  const seqGroups: BatchAssembly['seqGroups'] = []
  const items: ComfyBatchItem[] = []

  const vidDir = path.join(root, 'comfy_input')
  if (fs.existsSync(vidDir)) {
    for (const name of fs.readdirSync(vidDir).sort()) {
      const p = path.join(vidDir, name)
      if (!fs.statSync(p).isFile()) continue
      items.push({ kind: 'video', stem: name.replace(/\.[^.]+$/, ''), name })
      inputs.push({ path: p, basename: name })
    }
  }
  const seqRoot = path.join(root, 'comfy_seq')
  if (fs.existsSync(seqRoot)) {
    for (const tag of fs.readdirSync(seqRoot).sort()) {
      const dir = path.join(seqRoot, tag)
      if (!fs.statSync(dir).isDirectory()) continue
      const seq = detectSequenceDir(dir)
      if (!seq) return { error: `folder "${tag}" has no recognized image sequence (EXR / PNG / JPG / TIFF)` }
      const sd = seq.kind === 'exr'
        ? { kind: 'exr' as const, pattern: `/input/seq/${tag}/${seq.prefix}${'#'.repeat(seq.pad)}${seq.ext}`, start: seq.start, end: seq.end, step: 1, colorspace: 'linear' }
        : { kind: 'img' as const, dir: `/input/seq/${tag}`, colorspace: 'as-is' }
      items.push({ kind: seq.kind, stem: tag, seq: sd })
      seqGroups.push({ subdir: tag, files: seq.files.map(f => ({ path: f, basename: path.basename(f) })) })
    }
  }
  if (items.length === 0) return { error: 'batch has no inputs — upload at least one clip or image-sequence folder' }
  return { inputs, seqGroups, items }
}

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status, headers: { 'Content-Type': 'application/json' },
  })
}

/** Apply one {node, path, value} to an API-format graph (object keyed by id). */
function applyPatch(graph: Record<string, unknown>, op: ComfyPatch) {
  const node = graph[op.node] as Record<string, unknown> | undefined
  if (!node) return
  const parts = op.path.split('.')
  let cur: Record<string, unknown> = node
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i]
    if (typeof cur[k] !== 'object' || cur[k] === null) cur[k] = {}
    cur = cur[k] as Record<string, unknown>
  }
  cur[parts[parts.length - 1]] = op.value
}

export async function POST(req: Request) {
  const session = await auth()
  if (!session?.user?.id) return jsonError('Unauthorized', 401)

  let body: Body
  try { body = await req.json() as Body } catch { return jsonError('invalid JSON body', 400) }

  const preset = loadComfyPreset(body.presetKey)
  if (!preset) return jsonError(`unknown preset: ${body.presetKey}`, 404)
  if (!preset.image || String(preset.image).startsWith('REPLACE')) {
    return jsonError(`preset ${preset.key} has no ComfyUI image configured`, 422)
  }

  // Resolve the staged input clip (if the form uploaded one).
  let inputPath: string | null = null
  let inputName: string | null = null
  if (body.stageId && body.inputName) {
    if (!/^[A-Za-z0-9_-]+$/.test(body.stageId)) return jsonError('bad stageId', 400)
    const base = path.basename(body.inputName)
    const p = path.join(os.tmpdir(), 'tunet-stages', body.stageId, 'comfy_input', base)
    if (!fs.existsSync(p)) {
      return jsonError(`input clip not found (stage may have expired — re-upload): ${base}`, 400)
    }
    inputPath = p
    inputName = base
  }

  // Resolve the SECONDARY staged input (face image / mask), if this is a
  // two-input preset and the form uploaded one (staged under comfy_input2).
  // (Two-input presets aren't batchable — batch presets declare no secondary.)
  const secondaryParam = comfySecondaryParam(preset)
  let secondary: { param: string; path: string; basename: string } | null = null
  if (body.stageId && body.secondaryName && secondaryParam) {
    if (!/^[A-Za-z0-9_-]+$/.test(body.stageId)) return jsonError('bad stageId', 400)
    const base = path.basename(body.secondaryName)
    const p = path.join(os.tmpdir(), 'tunet-stages', body.stageId, 'comfy_input2', base)
    if (!fs.existsSync(p)) {
      return jsonError(`${secondaryParam} file not found (stage may have expired — re-upload): ${base}`, 400)
    }
    secondary = { param: secondaryParam, path: p, basename: base }
  } else if (secondaryParam && !body.secondaryName && !body.batch) {
    return jsonError(`preset ${preset.key} needs a ${secondaryParam} file (e.g. a face reference image)`, 422)
  }

  // BATCH: discover every staged input (videos + image-sequence folders) up front.
  let batch: BatchAssembly | null = null
  if (body.batch) {
    if (!body.stageId || !/^[A-Za-z0-9_-]+$/.test(body.stageId)) return jsonError('bad stageId', 400)
    const res = assembleBatch(body.stageId)
    if ('error' in res) return jsonError(res.error, 400)
    batch = res
  }

  const t0       = Date.now()
  const encoder  = new TextEncoder()
  const stageRoot = body.stageId
    ? path.join(os.tmpdir(), 'tunet-stages', body.stageId)
    : null

  const stream = new ReadableStream({
    async start(controller) {
      const send = (e: Record<string, unknown>) =>
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(e)}\n\n`))
      const heartbeat = setInterval(() => {
        try { controller.enqueue(encoder.encode(': keepalive\n\n')) } catch { /* closed */ }
      }, 15_000)

      let tarballPath: string | null = null
      try {
        send({ phase: 'validate', status: 'done' })

        // ── Pack ────────────────────────────────────────────────────────
        send({ phase: 'pack', status: 'start' })
        const tPack    = Date.now()
        const workflow = loadComfyWorkflow(preset)
        // For a batch, the per-input clip + output-prefix patches are applied per
        // item on the node (run_batch), so build only the GLOBAL knob patches here
        // (inputBasename=null skips the video + output_prefix ops).
        const patches  = buildComfyPatches(
          preset, body.values ?? {}, batch ? null : inputName,
          batch ? null : (secondary ? { param: secondary.param, basename: secondary.basename } : null),
        )
        // UI-format graphs (our presets) carry a top-level `nodes` array and are
        // converted to API on the node, so patches ride in patches.json. API
        // graphs are patched here.
        const isUi = !!(workflow as { nodes?: unknown }).nodes
        if (!isUi) {
          for (const op of patches) applyPatch(workflow as Record<string, unknown>, op)
        }
        const batchBuilt = batch ? buildComfyBatchManifest(preset, batch.items) : null
        const inputs = batch
          ? batch.inputs
          : [
              inputPath && inputName ? { path: inputPath, basename: inputName } : null,
              secondary ? { path: secondary.path, basename: secondary.basename } : null,
            ].filter((x): x is { path: string; basename: string } => x !== null)
        const pack = await packComfyTarball({
          workflow,
          patches: isUi ? patches : null,
          inputs,
          seqGroups: batch?.seqGroups,
        })
        tarballPath = pack.tarballPath
        send({ phase: 'pack', status: 'done', files: pack.fileCount, kb: Math.round(pack.compressedSize / 1024), ms: Date.now() - tPack })

        // ── Submit ──────────────────────────────────────────────────────
        send({ phase: 'submit', status: 'start' })
        const tSubmit = Date.now()
        // comfy_run.py self-uploads /output to ShareSync after rendering using
        // this bearer, so it must outlive the render. (See note in lib/comfy.ts.)
        const uploadToken = await getToken()
        const refreshCtx  = await getRefreshContext()
        const env = buildComfyEnv(preset, uploadToken, {
          hasPatches:   isUi && patches.length > 0,
          loras:        body.loras,
          readyTimeout: body.readyTimeout,
          refresh:      refreshCtx,
          batch:        batchBuilt,
        })
        const rawName = typeof body.values?.['__name'] === 'string' ? (body.values['__name'] as string).trim() : ''
        const jobName = rawName || `comfy-${preset.key}`
        // Spark doesn't echo the submit `name` back in list/detail responses, so
        // stash it (+ preset/mode markers) in env for the dashboard to show —
        // same as training jobs. Without this the dashboard falls back to the id.
        env.TUNET_JOB_NAME = jobName
        env.TUNET_PRESET   = preset.key
        env.TUNET_MODE     = 'comfy'
        const mode = body.mode === 'smart' ? 'smart' : 'instant'
        // Stash the form values (prompt + knobs) so the job page can show what
        // this render used and the Clone button can rehydrate the form. The clip
        // is NOT stashed — it's re-picked on clone (like the training dataset).
        // Mirrors training's TUNET_FORM_STATE; discriminated by `presetKey`.
        const stashValues: Record<string, unknown> = { ...(body.values ?? {}) }
        delete stashValues.__name
        env.TUNET_FORM_STATE = JSON.stringify(buildComfyFormState({
          presetKey: preset.key,
          gpu:       body.gpu ?? preset.gpu ?? 'rtxpro6000',
          mode,
          name:      rawName,
          values:    stashValues,
        }))
        const submitResp = await submitJob({
          name:            jobName,
          instanceType:    comfyInstanceType(preset, body.gpu),
          image:           preset.image,
          command:         comfyCommand(preset),
          mode,
          idleHoldSeconds: body.idleHoldSeconds ?? 0,
          tags:            ['cpfx_comfy'],
          env,
          // Read-only, node-cached model library (v1.23 §3.5): when an operator
          // has staged the ComfyUI models tree on ShareSync, comfy_run symlinks
          // weights from /assets instead of re-downloading them from HF/CivitAI.
          assetsShareSyncPath:      process.env.SPARK_COMFY_ASSETS_PATH || undefined,
          assetsShareSyncSpaceName: process.env.SPARK_COMFY_ASSETS_SPACE || undefined,
          // Comfy images are 10-20 GB — ask the scheduler to prefer a node that
          // already cached the image and report cold-pull vs hit (never delays).
          imageAffinity: 'required',
        })
        send({ phase: 'submit', status: 'done', jobId: submitResp.jobId, ms: Date.now() - tSubmit })

        if (submitResp.output?.shareSyncBaseUrl) {
          writeDiscoveredFilesBase(
            submitResp.output.shareSyncBaseUrl,
            submitResp.output.shareSyncPath ?? submitResp.outputShareSyncPath,
          )
        }

        // ── Upload ──────────────────────────────────────────────────────
        send({ phase: 'upload', status: 'start', totalBytes: pack.compressedSize })
        const tUpload = Date.now()
        await uploadInputTarball(submitResp.input.uploadUrl, pack.tarballPath)
        send({ phase: 'upload', status: 'done', sentBytes: pack.compressedSize, totalBytes: pack.compressedSize, ms: Date.now() - tUpload })

        // Clean the staged input clip now that it's shipped.
        if (stageRoot) await fs.promises.rm(stageRoot, { recursive: true, force: true }).catch(() => {})

        send({ phase: 'done', jobId: submitResp.jobId, output: submitResp.output, totalMs: Date.now() - t0 })
      } catch (e) {
        send({ phase: 'error', error: e instanceof Error ? e.message : String(e), totalMs: Date.now() - t0 })
      } finally {
        clearInterval(heartbeat)
        if (tarballPath) await fs.promises.rm(tarballPath, { force: true }).catch(() => {})
        controller.close()
      }
    },
  })

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type':      'text/event-stream',
      'Cache-Control':     'no-cache, no-transform',
      'Connection':        'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  })
}
