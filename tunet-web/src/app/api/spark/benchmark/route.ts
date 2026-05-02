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
// 45 sequential (pack + submit + upload) cycles. Tarball pack alone for a
// 350 MB dataset takes ~5 s; submit ~2 s; upload to Spark ~10–30 s depending
// on bandwidth. Budget ~30 s/cell so 45 cells fit under 800 s.
export const maxDuration = 800

// ── Reference benchmark settings ─────────────────────────────────────────────
// resolution + batch_size are now per-job axes (cartesian-product across the
// `resolutions` and `batchSizes` request fields). Everything else stays
// fixed so the only variables are GPU × res × batch.
const BENCH_FIXED = {
  model_size_dims: 64,
  model_type:      'unet' as const,
  loss:            'l1' as const,
  // No augmentation, no progressive res — pure throughput measurement.
  overlap_factor:  0.25,
} as const

// Default GPUs to benchmark — single-GPU SKUs only, matches the new-job UI.
const DEFAULT_GPU_KEYS: GpuKey[] = ['t4', 'l4', 'a10', 'l40s', 'rtxpro6000']
// Default sweep grid. Skips batch=1 (always sublinear) and batch=16
// (almost certainly OOM on the smaller cards at high res).
const DEFAULT_RESOLUTIONS = [256, 512, 1024]
const DEFAULT_BATCH_SIZES = [2, 4, 8]

// Synthetic dataset: 8 pairs at 1280×1280 so even the 1024px patch grid
// produces multiple valid extracts per image (ensures the dataloader
// doesn't run dry).
const SYNTH_PAIRS = 8
const SYNTH_SIZE  = 1280

interface Body {
  gpus?:            string[]
  /** Resolutions to sweep (px). Default [256, 512, 1024]. */
  resolutions?:     number[]
  /** Batch sizes to sweep. Default [2, 4, 8]. */
  batchSizes?:      number[]
  benchmarkSteps?:  number
  benchmarkWarmup?: number
  /**
   * Optional pre-uploaded stage from /api/spark/upload-stage. When present,
   * the benchmark uses the user's real frames (more realistic dataloader I/O
   * than the synthetic PNGs) so the calibrated baselineStepsPerSec matches
   * what production runs see. Must contain at least src/ and dst/.
   */
  stageId?:         string
  /**
   * Extra env vars to set on every submitted cell. Used for A/B-style
   * experiments — e.g. submit one cell with `{TUNET_DISABLE_COMPILE: '1'}`
   * and one without to measure torch.compile's impact. We deliberately
   * keep this opaque (no allow-list) since it's an admin/dev surface.
   */
  extraEnv?:        Record<string, string>
  /**
   * Optional name suffix appended to the per-cell job name. Lets A/B
   * runs distinguish the same (gpu, res, batch) cell run twice with
   * different env. e.g. nameSuffix='compile-on' → bench-l4-512px-bs2-compile-on-{stamp}
   */
  nameSuffix?:      string
}

interface Run {
  gpuKey:     string
  sku:        string
  jobId:      string
  label:      string
  resolution: number
  batchSize:  number
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

  // Resolution + batch-size sweep axes. Validate ranges so a fat-fingered
  // submit doesn't fire 500 jobs at 4096px.
  const resolutions = (body.resolutions && body.resolutions.length > 0) ? body.resolutions : DEFAULT_RESOLUTIONS
  const batchSizes  = (body.batchSizes  && body.batchSizes.length  > 0) ? body.batchSizes  : DEFAULT_BATCH_SIZES
  for (const r of resolutions) {
    if (!Number.isInteger(r) || r < 64 || r > 2048) return jsonError(`resolution ${r} out of range [64, 2048]`, 400)
  }
  for (const b of batchSizes) {
    if (!Number.isInteger(b) || b < 1 || b > 64) return jsonError(`batch size ${b} out of range [1, 64]`, 400)
  }
  // Cap the cartesian product so a misclick can't fire 100+ jobs at once.
  const totalJobs = gpuKeys.length * resolutions.length * batchSizes.length
  if (totalJobs > 60) {
    return jsonError(`would submit ${totalJobs} jobs (cap is 60). Trim the GPU/resolution/batch grid.`, 400)
  }

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

  // ── 2. Submit one job per (gpu × resolution × batch) cell ─────────────────
  // Each cell needs a unique tarball (config has the cell's res + batch
  // baked in), but the staged dataset is shared. With 45 cells and a
  // ~350 MB dataset, doing this serially blows past any reasonable HTTP
  // timeout (~25 min wall). We pipeline with bounded concurrency so the
  // server can pack one tarball while it uploads another.
  const runs: Run[] = []
  const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)
  const filesBase = process.env.SPARK_FILES_BASE_URL?.replace(/\/+$/, '')

  // Build the full work list first so we can iterate it in a worker pool.
  const cells: { gpuKey: GpuKey; sku: string; resolution: number; batchSize: number }[] = []
  for (const gpuKey of gpuKeys) {
    const sku = GPU_TYPES[gpuKey].sku
    for (const resolution of resolutions) {
      for (const batchSize of batchSizes) {
        cells.push({ gpuKey, sku, resolution, batchSize })
      }
    }
  }

  async function submitOne(cell: typeof cells[0]): Promise<Run> {
    const suffix = body.nameSuffix ? `-${body.nameSuffix}` : ''
    const jobName = `bench-${cell.gpuKey}-${cell.resolution}px-bs${cell.batchSize}${suffix}-${stamp}`
    const outputDir = `/output/${jobName}`

    const cfg = buildBenchmarkConfig(cell.resolution, cell.batchSize)
    const data = cfg.data as Record<string, unknown>
    const pack = await packInputTarball({
      config:   { ...cfg, data: { ...data, output_dir: outputDir } },
      stageDir,
    })

    const submitResp = await submitJob({
      name:            jobName,
      instanceType:    cell.sku,
      image:           DEFAULT_IMAGE,
      command:         ['bash', '/input/spark_start.sh', outputDir, '/input/config.yaml'],
      idleHoldSeconds: 0,
      env: {
        TUNET_JOB_NAME:    jobName,
        TUNET_PRESET:      'benchmark',
        TUNET_GPU:         cell.gpuKey,
        TUNET_BENCH_RES:   String(cell.resolution),
        TUNET_BENCH_BATCH: String(cell.batchSize),
        BENCHMARK_STEPS:   String(benchmarkSteps),
        BENCHMARK_WARMUP:  String(benchmarkWarmup),
        ...(filesBase ? { TUNET_FILES_BASE: filesBase } : {}),
        // Caller-supplied extras spread last so they can override anything
        // above if needed (e.g. tweak BENCHMARK_STEPS per-cell).
        ...(body.extraEnv ?? {}),
      },
    })

    await uploadInputTarball(submitResp.input.uploadUrl, pack.buffer)

    if (submitResp.output?.shareSyncBaseUrl) {
      try {
        writeDiscoveredFilesBase(submitResp.output.shareSyncBaseUrl, submitResp.output.shareSyncPath)
      } catch { /* ignore */ }
    }

    return {
      gpuKey:     cell.gpuKey,
      sku:        cell.sku,
      jobId:      submitResp.jobId,
      label:      GPU_TYPES[cell.gpuKey].label,
      resolution: cell.resolution,
      batchSize:  cell.batchSize,
    }
  }

  // Sequential submits. We tried concurrency=4 but it tripped Windows
  // EBUSY: parallel tarball packs all `copyFile` from the same tunet repo
  // source files, and Windows locks the source for the duration of the
  // copy. Serial keeps things simple and the wall-clock penalty is small
  // compared to Spark's submit + tarball upload time anyway.
  //
  // (If this becomes a bottleneck: build a single `tunet/` mirror up
  // front, then do per-cell pack from that mirror in parallel.)
  try {
    for (const cell of cells) {
      const r = await submitOne(cell)
      runs.push(r)
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
function buildBenchmarkConfig(resolution: number, batchSize: number): Record<string, unknown> {
  return {
    data: {
      src_dir:        '/input/data/src',
      dst_dir:        '/input/data/dst',
      output_dir:     '/output/_bench',     // overridden per-job above
      resolution:     resolution,
      overlap_factor: BENCH_FIXED.overlap_factor,
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
      model_type:       BENCH_FIXED.model_type,
      model_size_dims:  BENCH_FIXED.model_size_dims,
      recurrence_steps: 2,
    },
    training: {
      loss:                    BENCH_FIXED.loss,
      lr:                      1e-4,
      lr_scheduler:            'none',
      lambda_lpips:            0.2,
      l1_weight:               0.5,
      l2_weight:               0.5,
      lpips_weight:            0.2,
      use_amp:                 true,
      batch_size:              batchSize,
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
