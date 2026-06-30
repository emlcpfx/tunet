/**
 * POST /api/spark/jobs/:id/export-onnx
 *
 * Fast path (auto-export already produced the file):
 *   If `<job>/exports/flame/*.onnx` exists and is newer than the latest
 *   .pth (within a small fuzz), just return its download URL — no new
 *   compute spawned. This is the common case now that training jobs
 *   default to `auto_export.interval = 10` (see spark-presets.ts).
 *
 * Slow path (no fresh export available, or no auto-export ran):
 *   Spin up a small CPU-bound Spark job that:
 *     1. Pulls the source job's latest .pth from ShareSync directly (via
 *        bearer in env — no need to round-trip the 800+ MB checkpoint
 *        through our tarball).
 *     2. Rebuilds the model from the checkpoint's bundled metadata and
 *        writes Flame ONNX + Nuke TorchScript into <output>/exports/.
 *     3. Overrides `outputShareSyncPath` to point at the *source* job's
 *        ShareSync dir, so the new files appear in the same Downloads
 *        panel as the original .pth.
 *
 * Returns either:
 *   { fromCache: true,  downloadUrl, checkpoint, exportName }      (fast)
 *   { fromCache: false, jobId, checkpoint, checkpointKB, packKB }  (slow)
 *
 * The export container uses the same DEFAULT_IMAGE as training (torch
 * 2.7.1 is pre-baked) — only onnx is pip-installed at runtime, so the
 * setup phase is ~15 s instead of the trainer's ~3-5 min.
 */

import {
  submitJob, getJob, listOutputDir, shareSyncBaseUrl, getToken,
  writeDiscoveredFilesBase, GPU_TYPES, DEFAULT_IMAGE,
  propfindDir, resolveFilesBase, runawayWallClockSeconds,
} from '@/lib/spark'
import * as fs from 'node:fs'
import { packInputTarball } from '@/lib/spark-packer'
import { uploadInputTarball } from '@/lib/spark'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 60

interface Body {
  /** Optional override — defaults to the latest checkpoint in the source dir. */
  checkpointName?: string
  /**
   * Which deliverable the caller wants back. The conversion job always writes
   * BOTH (flame .onnx + nuke .pt/.nk/.cat) — this only selects which one the
   * fast-path cache check looks for and which file we name in the response.
   * Defaults to 'flame' for back-compat with the old single "Export now" button.
   */
  format?: 'flame' | 'nuke'
  /**
   * Skip the cache fast-path and always spawn a fresh conversion of the picked
   * checkpoint. The Downloads panel sets this because the user explicitly picks
   * a .pth and expects *that* one exported (the cache returns the newest export,
   * which may be a different epoch).
   */
  force?: boolean
}

// Per-format: which exports/ subdir to scan and how to pick the primary file.
//   flame → the .onnx (Flame / AE deliverable)
//   nuke  → prefer the .cat (Cattery model), else the .pt (TorchScript)
const FORMAT_SPEC = {
  flame: { dir: 'flame', pick: (es: { name: string }[]) => es.filter(e => e.name.endsWith('.onnx')) },
  nuke:  { dir: 'nuke',  pick: (es: { name: string }[]) => {
    const cat = es.filter(e => e.name.endsWith('.cat'))
    return cat.length ? cat : es.filter(e => e.name.endsWith('.pt'))
  } },
} as const

export async function POST(
  req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params

  let body: Body = {}
  try { body = await req.json() } catch { /* empty body is fine */ }
  const format: 'flame' | 'nuke' = body.format === 'nuke' ? 'nuke' : 'flame'

  const source = await getJob(id)
  if (!source) return jsonError('source job not found', 404)
  if (!source.output_share_sync_path) {
    return jsonError('source job has no output_share_sync_path', 422)
  }
  await resolveFilesBase()   // authoritative ShareSync base (v1.23 §3.4) before the sync guard
  if (!shareSyncBaseUrl(source)) {
    return jsonError(
      'source job has no resolvable ShareSync base URL — fix SPARK_FILES_BASE_URL in .env.local',
      503,
    )
  }

  // ── Discover the .pth to export ──────────────────────────────────────────
  // Prefer the user-provided name; otherwise pick the latest .pth listed in
  // the source output dir (matches the same heuristic the Downloads panel
  // uses: `_latest.pth` wins, else the most-recently-modified .pth).
  let entries
  try {
    entries = await listOutputDir(source)
  } catch (e) {
    return jsonError(
      `cannot list source output dir: ${e instanceof Error ? e.message : 'unknown'}`,
      502,
    )
  }
  const pthEntries = entries.filter(e => !e.isDir && e.name.endsWith('.pth'))
  if (pthEntries.length === 0) {
    return jsonError(
      'source job has no .pth checkpoints yet — wait for the first epoch to finish',
      409,
    )
  }
  const picked = body.checkpointName
    ? pthEntries.find(e => e.name === body.checkpointName)
    : pickLatest(pthEntries)
  if (!picked) {
    return jsonError(
      `checkpoint ${body.checkpointName} not found in source output`,
      404,
    )
  }

  // ── Build the WebDAV URL for the picked .pth ─────────────────────────────
  // shareSyncBaseUrl() returns the per-job-subdir URL; appending the file
  // basename gives us a full PROPFIND/GET-able URL. The export container
  // hits this with the bearer we hand it in env.
  const base = shareSyncBaseUrl(source)
  if (!base) return jsonError('no resolvable base URL', 503)
  const checkpointUrl = `${base}/${encodeURIComponent(picked.name)}`

  // ── Fast path: did auto_export already produce the requested deliverable? ─
  // Training jobs default to `auto_export.interval = 10`, so a job may have an
  // export waiting in <base>/exports/<flame|nuke>/. We hand back the latest one
  // — the filename embeds the epoch (e.g. `myrun_epoch_0040_052026_1530.onnx`)
  // so the user can read what they got. We skip this entirely when `force` is
  // set: the Downloads panel forces a fresh convert because the user explicitly
  // picks a .pth and expects *that* epoch, not whatever's newest on disk.
  // `downloadPath` is the job-relative path the client feeds to the files proxy
  // (the raw ShareSync URL needs a bearer the browser doesn't have). Best-effort:
  // any error here just falls through to the slow path.
  const spec = FORMAT_SPEC[format]
  if (!body.force) {
    try {
      const subEntries = await propfindDir(`${base}/exports/${spec.dir}`)
      const candidates = spec.pick(subEntries.filter(e => !e.isDir))
      const latest = pickLatest(candidates as PthEntry[])
      if (latest) {
        return new Response(JSON.stringify({
          fromCache:    true,
          format,
          checkpoint:   picked.name,
          exportName:   latest.name,
          downloadPath: `exports/${spec.dir}/${latest.name}`,
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }
    } catch {
      // Sub-dir PROPFIND failed (network, auth, malformed XML). Don't penalize
      // the user — fall through to the slow path and they get the same outcome
      // they would have gotten before this fast-path existed.
    }
  }

  // ── Recover the per-job subdir so the export job writes to the same place ─
  // The source's command[2] is `/output/<safeName>`; train.py wrote the .pth
  // into that subdir. We pass the same /output/<safeName> as the export
  // container's $1 so the exports/ tree lands in the same WebDAV folder.
  const sourceSubdir = jobOutputSubdir(source)
  if (!sourceSubdir) {
    return jsonError(
      'cannot recover source job\'s output subdir from command',
      422,
    )
  }
  const exportOutputDir = `/output/${sourceSubdir}`

  // ── Pack tiny tarball (no .pth, no data) ─────────────────────────────────
  // The packer's normal excludes already drop .pth/.onnx/__pycache__/etc.
  // We pass an empty config (the export script ignores it — config lives
  // inside the .pth) and override the entrypoint script name.
  let pack
  try {
    pack = await packInputTarball({
      config: {},
      startScriptName:       'spark_export.sh',
      startScriptSourceName: 'spark_export.sh',
    })
  } catch (e) {
    return jsonError(`pack failed: ${e instanceof Error ? e.message : 'unknown'}`, 500)
  }

  // ── Submit ───────────────────────────────────────────────────────────────
  const token = await getToken()
  const safeName = `export_${sourceSubdir}_${Date.now().toString(36)}`.slice(0, 64)
  const filesBase = process.env.SPARK_FILES_BASE_URL?.replace(/\/+$/, '')

  let submitResp
  try {
    submitResp = await submitJob({
      name:               `Export ONNX: ${source.env?.TUNET_JOB_NAME ?? sourceSubdir}`,
      instanceType:       GPU_TYPES.t4.sku,           // cheapest — export is CPU-bound anyway
      image:              DEFAULT_IMAGE,
      command:            ['bash', '/input/spark_export.sh', exportOutputDir, '/input/config.yaml'],
      idleHoldSeconds:    0,
      mode:               'smart',                    // ~60% cheaper, preempt is fine
      maxRetriesOnInterrupt: 2,
      maxWallClockSeconds: runawayWallClockSeconds(),  // billing backstop (export is bounded)
      // Write back into the source job's ShareSync dir so the resulting
      // exports/ tree shows up alongside the .pth in the same Downloads
      // panel — no separate page needed.
      outputShareSyncPath: source.output_share_sync_path,
      env: {
        TUNET_JOB_NAME:  safeName,
        TUNET_MODE:      'export-onnx',
        TUNET_SOURCE_JOB_ID: source.id,
        // Spark rejects env keys with the SPARK_ prefix (reserved). Use a
        // neutral name; spark_export.sh reads the same value.
        TUNET_BEARER:    token,                       // short-lived; expires < 1h
        CHECKPOINT_URL:  checkpointUrl,
        CKPT_PREFIX:     sourceSubdir,
        ...(filesBase ? { TUNET_FILES_BASE: filesBase } : {}),
      },
    })
  } catch (e) {
    return jsonError(`submit failed: ${e instanceof Error ? e.message : 'unknown'}`, 502)
  }

  // Cache the discovered base URL if we got one (matches training submit).
  if (submitResp.output?.shareSyncBaseUrl) {
    writeDiscoveredFilesBase(
      submitResp.output.shareSyncBaseUrl,
      submitResp.output.shareSyncPath ?? submitResp.outputShareSyncPath,
    )
  }

  // ── Upload tarball ───────────────────────────────────────────────────────
  try {
    await uploadInputTarball(submitResp.input.uploadUrl, pack.tarballPath)
  } catch (e) {
    return jsonError(`upload failed: ${e instanceof Error ? e.message : 'unknown'}`, 502)
  } finally {
    await fs.promises.rm(pack.tarballPath, { force: true }).catch(() => {})
  }

  return new Response(JSON.stringify({
    fromCache:    false,
    format,
    // Where the deliverable will land once the job finishes — the client polls
    // the files listing for new entries under this dir and auto-downloads.
    exportDir:    `exports/${FORMAT_SPEC[format].dir}`,
    jobId:        submitResp.jobId,
    checkpoint:   picked.name,
    checkpointKB: Math.round(picked.size / 1024),
    packKB:       Math.round(pack.compressedSize / 1024),
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

interface PthEntry { name: string; size: number; modified: string | null }

function pickLatest(entries: PthEntry[]): PthEntry | undefined {
  const latest = entries.find(e => /_latest\.pth$/.test(e.name))
  if (latest) return latest
  return entries.slice().sort((a, b) => {
    const ta = a.modified ? Date.parse(a.modified) : 0
    const tb = b.modified ? Date.parse(b.modified) : 0
    return tb - ta
  })[0]
}

function jobOutputSubdir(j: { command?: string[] }): string | null {
  const cmd = j.command
  if (!Array.isArray(cmd) || cmd.length < 3) return null
  const arg = cmd[2]
  if (typeof arg !== 'string' || !arg.startsWith('/output/')) return null
  return arg.slice('/output/'.length).split('/').filter(Boolean)[0] ?? null
}
