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
} from '@/lib/spark'
import { packInputTarball } from '@/lib/spark-packer'
import {
  PRESETS, buildConfig, type PresetKey, type AdvancedOverrides, type JobInputs,
} from '@/lib/spark-presets'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 60

interface Body {
  name?:    string
  preset?:  string
  gpu?:     string
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
   * Optional: a stageId from /api/spark/upload-stage. If present, the staged
   * data folders (src/, dst/, val_src/, val_dst/) are bundled into the
   * tarball under data/<role>/ and the config is rewritten to point at
   * /input/data/<role>/. Overrides any inputs.src_dir / inputs.dst_dir.
   */
  stageId?: string
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
  const outputDir = body.inputs?.output_dir ?? `/output/${safeName}`

  const jobInputs: JobInputs = stageDir ? {
    // When data is staged in the tarball, paths are deterministic
    src_dir:       '/input/data/src',
    dst_dir:       '/input/data/dst',
    output_dir:    outputDir,
    val_src_dir:   stageRoles.has('val_src') ? '/input/data/val_src' : null,
    val_dst_dir:   stageRoles.has('val_dst') ? '/input/data/val_dst' : null,
    finetune_from: body.inputs?.finetune_from ?? null,
  } : {
    src_dir:       body.inputs!.src_dir!,
    dst_dir:       body.inputs!.dst_dir!,
    output_dir:    outputDir,
    val_src_dir:   body.inputs?.val_src_dir   ?? null,
    val_dst_dir:   body.inputs?.val_dst_dir   ?? null,
    finetune_from: body.inputs?.finetune_from ?? null,
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
        const pack = await packInputTarball({ config, stageDir: stageDir ?? undefined })
        send({
          phase:  'pack',
          status: 'done',
          files:  pack.fileCount,
          kb:     Math.round(pack.compressedSize / 1024),
          ms:     Date.now() - tPack,
        })

        // ── Submit ──────────────────────────────────────────────────────
        send({ phase: 'submit', status: 'start' })
        const tSubmit = Date.now()
        const submitResp = await submitJob({
          name,
          instanceType:    sku,
          image:           DEFAULT_IMAGE,
          command:         ['bash', '/input/spark_start.sh', outputDir, '/input/config.yaml'],
          idleHoldSeconds: body.idleHoldSeconds ?? 0,
          // Spark doesn't echo the submit `name` back in list/detail responses,
          // so we stash the friendly name + preset/gpu in env. The container
          // ignores these (they're just metadata for our own UI to read back).
          env: {
            TUNET_JOB_NAME: name,
            TUNET_PRESET:   preset.key,
            TUNET_GPU:      gpuKey,
          },
        })
        send({
          phase:  'submit',
          status: 'done',
          jobId:  submitResp.jobId,
          ms:     Date.now() - tSubmit,
        })

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
