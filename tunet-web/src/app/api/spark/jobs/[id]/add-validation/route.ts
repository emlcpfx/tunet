/**
 * POST /api/spark/jobs/:id/add-validation
 *
 * Adds validation data to an ALREADY-RUNNING training job. Validation data is
 * normally baked into the job tarball at submit; once a job is running the only
 * channel to it is ShareSync, so we deliver new frames through the same
 * `_tunet_control/<key>/` control area that request-export / set-max-steps use.
 *
 *   • multipart batch (fields: batchId, role ∈ val_src|val_dst, files[]) — PUTs
 *     each file to `_tunet_control/<key>/val_add/<batchId>/<role>/<basename>`.
 *     The client uploads in batches (browser FormData has practical size limits).
 *
 *   • ?commit=1 with JSON { batchId, src:[names], dst:[names] } — writes
 *     `validation_request.json` (nonce + manifest). train.py polls it (same
 *     cadence as the export channel, rank 0 only), downloads the listed files,
 *     ADDS them to its current validation set, and rebuilds the val dataloader.
 *
 * Only meaningful while the job is running — a finished job has nothing to poll.
 */
import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { getJob } from '@/lib/spark'
import { putControlFile, putControlBytes } from '@/lib/sharesync'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
// Validation batches are small (a handful of frames), but EXRs add up — give
// the upload room like the staging route does.
export const maxDuration = 300

const ROLES = new Set(['val_src', 'val_dst'])

/** Output subdir from the job command: ['bash', spark_start.sh, /output/<sub>, ...]. */
function controlKey(job: { command?: unknown }): string | null {
  const cmd = job.command
  if (!Array.isArray(cmd)) return null
  const arg = cmd.find(a => typeof a === 'string' && a.startsWith('/output/')) as string | undefined
  if (!arg) return null
  return arg.slice('/output/'.length).split('/').filter(Boolean)[0] ?? null
}

/** Keep batch ids to a safe, single-segment slug (no traversal, no spaces). */
function sanitizeBatchId(raw: string): string {
  return raw.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64)
}

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const session = await auth()
  if (!session?.user?.id) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { id } = await ctx.params

  let job
  try { job = await getJob(id) } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : 'failed' }, { status: 502 })
  }
  if (!job) return NextResponse.json({ error: 'job not found' }, { status: 404 })

  const key = controlKey(job)
  if (!key) return NextResponse.json({ error: 'cannot derive output subdir for this job' }, { status: 422 })

  const status = String(job.status ?? '').toLowerCase()
  const live = !['succeeded', 'completed', 'failed', 'cancelled', 'stopped', 'error'].includes(status)
  if (!live) {
    return NextResponse.json({
      error: `job is '${job.status}', not running — there's nothing to add validation data to. Add it when creating the job, or clone and resubmit.`,
      notLive: true,
    }, { status: 409 })
  }

  // ── Commit phase: write the manifest the trainer polls ────────────────────
  const commit = new URL(req.url).searchParams.get('commit')
  if (commit) {
    let body: { batchId?: unknown; src?: unknown; dst?: unknown }
    try { body = await req.json() } catch { return NextResponse.json({ error: 'expected JSON body' }, { status: 400 }) }
    const batchId = sanitizeBatchId(String(body.batchId ?? ''))
    const src = Array.isArray(body.src) ? body.src.map(String) : []
    const dst = Array.isArray(body.dst) ? body.dst.map(String) : []
    if (!batchId) return NextResponse.json({ error: 'batchId required' }, { status: 400 })
    if (src.length === 0) return NextResponse.json({ error: 'need at least one val_src file' }, { status: 400 })
    try {
      await putControlFile(key, 'validation_request.json', { nonce: Date.now(), batchId, src, dst })
    } catch (e) {
      return NextResponse.json({ error: e instanceof Error ? e.message : 'failed to write request' }, { status: 502 })
    }
    return NextResponse.json({
      committed: { batchId, src: src.length, dst: dst.length },
      note: dst.length > 0
        ? 'the running job will add these to its validation set (loss + previews) at its next checkpoint'
        : 'the running job will add these as validation previews (no dst → no loss metrics) at its next checkpoint',
    })
  }

  // ── Upload phase: a single batch of files for one role ────────────────────
  let form: FormData
  try { form = await req.formData() } catch (e) {
    return NextResponse.json({ error: 'expected multipart/form-data', detail: e instanceof Error ? e.message : '' }, { status: 400 })
  }

  const batchId = sanitizeBatchId(String(form.get('batchId') ?? ''))
  if (!batchId) return NextResponse.json({ error: 'batchId required' }, { status: 400 })

  const role = String(form.get('role') ?? '')
  if (!ROLES.has(role)) return NextResponse.json({ error: `role must be one of: ${[...ROLES].join(', ')}` }, { status: 400 })

  const files = form.getAll('files').filter(f => f instanceof File) as File[]
  if (files.length === 0) return NextResponse.json({ error: 'no files in request' }, { status: 400 })

  const names: string[] = []
  let bytes = 0
  for (const file of files) {
    // basename only — the browser sends webkitRelativePath like 'proj/val_src/f.exr'
    const safeName = file.name.split(/[\\/]/).pop() ?? ''
    if (!safeName || safeName.startsWith('.')) continue
    const buf = await file.arrayBuffer()
    try {
      await putControlBytes(key, ['val_add', batchId, role, safeName], buf)
    } catch (e) {
      return NextResponse.json({ error: e instanceof Error ? e.message : 'upload failed', file: safeName }, { status: 502 })
    }
    names.push(safeName)
    bytes += buf.byteLength
  }

  return NextResponse.json({ batchId, role, uploaded: names.length, bytes, names }, { status: 201 })
}
