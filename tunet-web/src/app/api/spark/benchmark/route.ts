/**
 * POST /api/spark/benchmark
 *
 * Submits one *benchmark* job per requested GPU at fixed reference settings
 * (UNet, model_size_dims=64, 512px, batch_size=2, L1 loss, no progressive
 * res, no augmentation). Each job runs ~200 steps after warmup, logs a
 * `STEP_RATE: X.Y step/sec` line, and exits.
 *
 * The point: feed real numbers back into baselineStepsPerSec() in
 * spark-presets.ts so the cost estimator stops being a pure guess.
 *
 * Body:
 *   {
 *     gpus?:           string[]  // GpuKey list; omit to run all 5 single-GPU SKUs
 *     benchmarkSteps?: number    // default 200
 *     benchmarkWarmup?: number   // default 20
 *   }
 *
 * Response:
 *   {
 *     runs: [{ gpuKey, sku, jobId, label }, ...]
 *   }
 *
 * Each run uses a shared synthetic dataset (8 procedurally-generated
 * 768×768 PNG pairs) bundled into the tarball so there's nothing for the
 * user to upload — same input across GPUs guarantees apples-to-apples
 * timing.
 *
 * Cost: ~1 minute of GPU time per run × $0.74–$6.50/hr ≈ $0.01–$0.11 per
 * GPU. All 5 ≈ $0.30. The web UI surfaces this estimate before submit.
 */

import 'server-only'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import * as crypto from 'node:crypto'
import sharp from 'sharp'
import { submitJob, uploadInputTarball, listSkus, GPU_TYPES, DEFAULT_IMAGE, type GpuKey, writeDiscoveredFilesBase } from '@/lib/spark'
import { packInputTarball } from '@/lib/spark-packer'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 120  // five sequential submits + five tarball packs

// ── Reference benchmark settings ─────────────────────────────────────────────
// These MUST stay in sync with the assumed baseline in
// spark-presets.ts → settingsMultiplier() (model_size 64, batch 2, UNet, L1)
// otherwise the calibration data won't apply 1:1 to the multiplier formula.
const BENCH_SETTINGS = {
  resolution:      512,
  model_size_dims: 64,
  model_type:      'unet' as const,
  batch_size:      2,
  loss:            'l1' as const,
  // No augmentation, no progressive res — pure throughput measurement.
  overlap_factor:  0.25,
} as const

// Default GPUs to benchmark — single-GPU SKUs only, matches the new-job UI.
const DEFAULT_GPU_KEYS: GpuKey[] = ['t4', 'l4', 'a10', 'l40s', 'rtxpro6000']

// Synthetic dataset: 8 pairs at 768×768 so the 512px patch grid produces
// usable patches without huge memory pressure.
const SYNTH_PAIRS = 8
const SYNTH_SIZE  = 768

interface Body {
  gpus?:            string[]
  benchmarkSteps?:  number
  benchmarkWarmup?: number
  /**
   * Optional pre-uploaded stage from /api/spark/upload-stage. When present,
   * the benchmark uses the user's real frames (more realistic dataloader I/O
   * than the synthetic PNGs) so the calibrated baselineStepsPerSec matches
   * what production runs see. Must contain at least src/ and dst/.
   */
  stageId?:         string
}

interface Run {
  gpuKey: string
  sku:    string
  jobId:  string
  label:  string
}

export async function POST(req: Request) {
  let body: Body
  try { body = await req.json() } catch { return jsonError('Invalid JSON', 400) }

  const requested = (body.gpus && body.gpus.length > 0)
    ? body.gpus
    : DEFAULT_GPU_KEYS
  const gpuKeys: GpuKey[] = []
  for (const k of requested) {
    if (!(k in GPU_TYPES)) return jsonError(`unknown gpu key: ${k}`, 400)
    gpuKeys.push(k as GpuKey)
  }

  // Spark's eligible-SKU allow-list changes (e.g. some regions don't have
  // L40S/g6e). Intersect what the user asked for with what's actually
  // allowed *now* so we get a clean error before submitting any jobs,
  // instead of an HTTP 400 mid-loop after one or two have already gone out.
  let allowedSkus: Set<string>
  try {
    const skus = await listSkus()
    allowedSkus = new Set(skus.map(s => s.instanceType))
  } catch (e) {
    return jsonError(`failed to fetch eligible SKUs: ${e instanceof Error ? e.message : e}`, 502)
  }
  const ineligible = gpuKeys.filter(k => !allowedSkus.has(GPU_TYPES[k].sku))
  if (ineligible.length > 0) {
    return jsonError(
      `Not eligible on this account: ${ineligible.map(k => `${k} (${GPU_TYPES[k].sku})`).join(', ')}. ` +
      `Eligible SKUs right now: ${[...allowedSkus].sort().join(', ')}`,
      400,
    )
  }
  const benchmarkSteps  = body.benchmarkSteps  && body.benchmarkSteps  > 0 ? body.benchmarkSteps  : 200
  const benchmarkWarmup = body.benchmarkWarmup && body.benchmarkWarmup > 0 ? body.benchmarkWarmup : 20

  // ── 1. Resolve the dataset stage dir ───────────────────────────────────────
  // Two paths:
  //   a) body.stageId set → user pre-uploaded a folder via /upload-stage.
  //      Use that dir as-is. More realistic dataloader I/O than synthetic.
  //   b) no stageId → generate 8 synthetic 768×768 PNG pairs into a fresh
  //      tmpdir. Cheap, deterministic, but PNG decode is much faster than
  //      EXR so the calibration ends up optimistic.
  // We track `cleanupSynthDir` separately so we never delete the user's
  // staged data on the way out.
  let stageDir:        string
  let cleanupSynthDir: string | null = null
  if (body.stageId) {
    stageDir = path.join(os.tmpdir(), 'tunet-stages', body.stageId)
    if (!fs.existsSync(stageDir)) {
      return jsonError(`stage not found: ${body.stageId}`, 404)
    }
    if (!fs.existsSync(path.join(stageDir, 'src')) || !fs.existsSync(path.join(stageDir, 'dst'))) {
      return jsonError(`stage ${body.stageId} must contain src/ and dst/ subfolders`, 400)
    }
  } else {
    const synthId = crypto.randomBytes(8).toString('base64url')
    stageDir = path.join(os.tmpdir(), 'tunet-benchmarks', synthId)
    cleanupSynthDir = stageDir
    await fs.promises.mkdir(path.join(stageDir, 'src'), { recursive: true })
    await fs.promises.mkdir(path.join(stageDir, 'dst'), { recursive: true })
    try {
      await generateSyntheticPairs(stageDir, SYNTH_PAIRS, SYNTH_SIZE)
    } catch (e) {
      return jsonError(`failed to generate synthetic dataset: ${e instanceof Error ? e.message : e}`, 500)
    }
  }

  // ── 2. Build the bench config (same for every GPU) ─────────────────────────
  const benchConfig = buildBenchmarkConfig()

  // ── 3. Submit one job per GPU sequentially ─────────────────────────────────
  // Sequential submits keep the rate-limit happy and let us bail early on a
  // bad token without leaving half the runs orphaned.
  const runs: Run[] = []
  const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)
  const filesBase = process.env.SPARK_FILES_BASE_URL?.replace(/\/+$/, '')

  try {
    for (const gpuKey of gpuKeys) {
      const sku = GPU_TYPES[gpuKey].sku
      const jobName = `bench-${gpuKey}-${stamp}`
      const outputDir = `/output/${jobName}`

      // Pack a fresh tarball per submit. Cheap (~800 KB) and avoids any
      // cross-job state confusion. Cast `data` to a plain record so we can
      // spread + override output_dir without losing the rest of the config.
      const data = benchConfig.data as Record<string, unknown>
      const pack = await packInputTarball({
        config:   { ...benchConfig, data: { ...data, output_dir: outputDir } },
        stageDir,
      })

      const submitResp = await submitJob({
        name:            jobName,
        instanceType:    sku,
        image:           DEFAULT_IMAGE,
        command:         ['bash', '/input/spark_start.sh', outputDir, '/input/config.yaml'],
        idleHoldSeconds: 0,
        env: {
          TUNET_JOB_NAME:   jobName,
          TUNET_PRESET:     'benchmark',
          TUNET_GPU:        gpuKey,
          BENCHMARK_STEPS:  String(benchmarkSteps),
          BENCHMARK_WARMUP: String(benchmarkWarmup),
          ...(filesBase ? { TUNET_FILES_BASE: filesBase } : {}),
        },
      })

      // Upload the tarball to the per-job upload URL.
      await uploadInputTarball(submitResp.input.uploadUrl, pack.buffer)

      // Cache discovered files-base on first successful submit.
      if (submitResp.output?.shareSyncBaseUrl) {
        try {
          writeDiscoveredFilesBase(submitResp.output.shareSyncBaseUrl, submitResp.output.shareSyncPath)
        } catch { /* ignore */ }
      }

      runs.push({
        gpuKey,
        sku,
        jobId: submitResp.jobId,
        label: GPU_TYPES[gpuKey].label,
      })
    }
  } catch (e) {
    return new Response(
      JSON.stringify({
        error: e instanceof Error ? e.message : String(e),
        runs,  // partial list so the UI can surface what succeeded
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  } finally {
    // Best-effort cleanup of synthetic dir only — never touch user-staged
    // data, the new-job page may still need it.
    if (cleanupSynthDir) {
      fs.promises.rm(cleanupSynthDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  return new Response(JSON.stringify({ runs }), {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

/**
 * Generate `count` deterministic src/dst pairs at `size`×`size`. Patterns:
 *   src = horizontal gradient + per-frame noise field
 *   dst = src with a small offset + tinted scribble (so the L1 loss has
 *         signal instead of converging to zero immediately)
 *
 * Determinism is intentional: same dataset across GPUs means the only
 * variable in measured step/sec is the hardware itself.
 */
async function generateSyntheticPairs(stageDir: string, count: number, size: number): Promise<void> {
  for (let i = 0; i < count; i++) {
    const srcPng = await synthFrame(size, i, /*tint*/ 0)
    const dstPng = await synthFrame(size, i, /*tint*/ 1)
    const idx = String(i).padStart(4, '0')
    await fs.promises.writeFile(path.join(stageDir, 'src', `frame_${idx}.png`), srcPng)
    await fs.promises.writeFile(path.join(stageDir, 'dst', `frame_${idx}.png`), dstPng)
  }
}

async function synthFrame(size: number, frameIdx: number, tint: 0 | 1): Promise<Buffer> {
  // Build raw RGB bytes. Pattern: gradient + checkerboard + per-frame phase.
  const data = Buffer.allocUnsafe(size * size * 3)
  const phase = (frameIdx * 31) & 0xff
  const tintR = tint ? 40 : 0
  const tintG = tint ? 0  : 0
  const tintB = tint ? 0  : 40
  let off = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const checker = ((x >> 5) + (y >> 5)) & 1 ? 80 : 30
      const grad    = Math.floor(((x + y + phase) / (2 * size)) * 200)
      data[off]     = clamp8(grad + checker + tintR)
      data[off + 1] = clamp8(grad + (checker >> 1) + tintG)
      data[off + 2] = clamp8((255 - grad) + (checker >> 2) + tintB)
      off += 3
    }
  }
  return sharp(data, { raw: { width: size, height: size, channels: 3 } })
    .png({ compressionLevel: 1 })   // fast — we don't care about size
    .toBuffer()
}

function clamp8(n: number): number {
  return n < 0 ? 0 : n > 255 ? 255 : n
}

/**
 * Build the YAML config dict that train.py will consume. Matches the shape
 * of base/base.yaml and what buildConfig() in spark-presets.ts produces for
 * a normal job — but stripped to just what the benchmark needs. Keep this
 * dependency-free of presets so the benchmark stays a stable measurement
 * point even if the preset library changes.
 */
function buildBenchmarkConfig(): Record<string, unknown> {
  return {
    data: {
      src_dir:        '/input/data/src',
      dst_dir:        '/input/data/dst',
      output_dir:     '/output/_bench',     // overridden per-job above
      resolution:     BENCH_SETTINGS.resolution,
      overlap_factor: BENCH_SETTINGS.overlap_factor,
      color_space:    'srgb',
      mask_dir:       null,
      val_src_dir:    null,
      val_dst_dir:    null,
    },
    mask: {
      use_mask_loss:        false,
      mask_weight:          10.0,
      use_mask_input:       false,
      use_auto_mask:        false,
      auto_mask_gamma:      1.0,
      skip_empty_patches:   false,
      skip_empty_threshold: 3.0,
    },
    model: {
      model_type:       BENCH_SETTINGS.model_type,
      model_size_dims:  BENCH_SETTINGS.model_size_dims,
      recurrence_steps: 2,
    },
    training: {
      loss:                    BENCH_SETTINGS.loss,
      lr:                      1e-4,
      lr_scheduler:            'none',
      lambda_lpips:            0.2,
      l1_weight:               0.5,
      l2_weight:               0.5,
      lpips_weight:            0.2,
      use_amp:                 true,
      batch_size:              BENCH_SETTINGS.batch_size,
      iterations_per_epoch:    1000,
      max_steps:               0,         // benchmark exits via --benchmark-steps
      progressive_resolution:  false,
    },
    augmentations: {},
    logging: {
      log_interval:           50,
      preview_batch_interval: 0,           // disable previews — pure throughput
      preview_refresh_rate:   0,
      val_interval:           0,
      diff_amplify:           1.0,
    },
    saving: {
      keep_last_checkpoints: 0,            // don't write checkpoints during bench
    },
    early_stopping: { enabled: false },
    auto_export: {
      auto_export_interval: 0,
      auto_export_flame:    false,
      auto_export_nuke:     false,
    },
    dataloader: {
      num_workers:     2,
      prefetch_factor: 2,
    },
  }
}
