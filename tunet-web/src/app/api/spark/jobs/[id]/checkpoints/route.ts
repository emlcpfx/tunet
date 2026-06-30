/**
 * GET /api/spark/jobs/:id/checkpoints
 *
 * Returns the list of `.pth` files in a prior job's output dir, used by the
 * Resume / Fine-tune pickers in /demo/jobs/new.
 *
 * Response:
 *   {
 *     job: { id, label, preset, gpuKey, outputSubdir },
 *     latest: { name, size, modified } | null,   // detected '*_tunet_latest.pth'
 *     epochs: { name, size, modified }[],         // newest-first, latest excluded
 *     pending?: true                              // output dir exists but no .pth yet
 *   }
 *
 * `latest` is what Resume needs (re-stage + train.py auto-resumes from it).
 * `epochs` plus `latest` is the full list a fine-tune picker can choose from.
 *
 * Auth via the server-side Spark bearer; the browser never sees the token.
 */

import { getJob, listOutputDir, shareSyncBaseUrl, resolveFilesBase } from '@/lib/spark'
import { jobLabel, jobPresetKey, jobGpuKey } from '@/lib/spark-types'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const PENDING_UPSTREAM_STATUSES = new Set([404, 500, 502, 503])

export async function GET(
  _req: Request,
  ctx:  { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params

  let job
  try {
    job = await getJob(id)
  } catch (e) {
    return jsonError(e instanceof Error ? e.message : 'failed', 502)
  }
  if (!job) return jsonError('job not found', 404)

  await resolveFilesBase()   // authoritative ShareSync base (v1.23 §3.4) before the sync guard
  if (!shareSyncBaseUrl(job)) {
    return jsonError(
      'ShareSync base URL not configured (set SPARK_FILES_BASE_URL in .env.local)',
      503,
    )
  }

  // Recover the per-job output subdir (basename of /output/<x> in command[2])
  // so the client can pass it back to /api/spark/training-jobs to keep the
  // resume run writing into the *same* dir.
  const outputSubdir = (() => {
    const cmd = job.command
    if (!Array.isArray(cmd) || cmd.length < 3) return null
    const arg = cmd[2]
    if (typeof arg !== 'string' || !arg.startsWith('/output/')) return null
    return arg.slice('/output/'.length).split('/').filter(Boolean)[0] ?? null
  })()

  let entries
  try {
    entries = await listOutputDir(job)
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'list failed'
    const m   = msg.match(/HTTP (\d+)/)
    const upstream = m ? parseInt(m[1], 10) : 0
    if (PENDING_UPSTREAM_STATUSES.has(upstream)) {
      return new Response(JSON.stringify({
        job:    { id: job.id, label: jobLabel(job), preset: jobPresetKey(job), gpuKey: jobGpuKey(job), outputSubdir },
        latest: null,
        epochs: [],
        pending: true,
      }), { headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } })
    }
    return jsonError(msg, 502)
  }

  const pthEntries = entries
    .filter(e => !e.isDir && e.name.endsWith('.pth'))
    .map(e => ({ name: e.name, size: e.size, modified: e.modified }))

  // The trainer always writes <ckpt_prefix>_tunet_latest.pth alongside epoch
  // checkpoints. Detect by suffix; fall back to mtime-newest if none matches.
  const byMtimeDesc = (a: { modified: string | null }, b: { modified: string | null }) =>
    (b.modified ? Date.parse(b.modified) : 0) - (a.modified ? Date.parse(a.modified) : 0)

  const latest =
    pthEntries.find(e => /_latest\.pth$/.test(e.name))
    ?? [...pthEntries].sort(byMtimeDesc)[0]
    ?? null

  const epochs = pthEntries
    .filter(e => latest === null || e.name !== latest.name)
    .sort(byMtimeDesc)

  return new Response(JSON.stringify({
    job: {
      id:           job.id,
      label:        jobLabel(job),
      preset:       jobPresetKey(job),
      gpuKey:       jobGpuKey(job),
      outputSubdir,
    },
    latest,
    epochs,
  }), {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  })
}

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
