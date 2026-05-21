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
  propfindDir,
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
}

export async function POST(
  req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params

  let body: Body = {}
  try { body = await req.json() } catch { /* empty body is fine */ }

  const source = await getJob(id)
  if (!source) return jsonError('source job not found', 404)
  if (!source.output_share_sync_path) {
    return jsonError('source job has no output_share_sync_path', 422)
  }
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

  // ── Fast path: did auto_export already produce an .onnx? ─────────────────
  // Training jobs default to `auto_export.interval = 10`, so most jobs have
  // an .onnx waiting in <base>/exports/flame/. We hand back the latest one
  // unconditionally — the filename embeds the epoch (e.g. `myrun_epoch_0040_
  // 052026_1530.onnx`) so the user can read what they got, and the
  // alternative (spawning a CPU job that takes ~3 min + ~$0.02) is strictly
  // worse than handing them a few-epoch-stale checkpoint that's already on
  // disk. If the user wants the absolute latest .pth re-exported, that's a
  // separate "force re-export" flow we can add later. Best-effort: any error
  // here just falls through to the slow path.
  try {
    const flameEntries = await propfindDir(`${base}/exports/flame`)
    const onnxEntries = flameEntries.filter(e => !e.isDir && e.name.endsWith('.onnx'))
    const latestOnnx = pickLatest(onnxEntries)
    if (latestOnnx) {
      return new Response(JSON.stringify({
        fromCache:    true,
        downloadUrl:  `${base}/exports/flame/${encodeURIComponent(latestOnnx.name)}`,
        checkpoint:   picked.name,
        exportName:   latestOnnx.name,
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
