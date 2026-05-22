/**
 * POST /api/spark/jobs/:id/set-max-steps   body: { maxSteps: number }
 *
 * Adjust a RUNNING training job's stop point live. Drops a `stop_request.json`
 * flag in the job's ShareSync control path; train.py polls it (same cadence as
 * the export channel) and updates its `max_steps` — stopping gracefully when
 * global_step reaches it (saving a final checkpoint), or near-immediately if the
 * value is already <= the current step. 0 = unlimited (un-cap the run).
 *
 * Only meaningful while the job is running; a finished job has nothing to poll.
 */
import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { getJob } from '@/lib/spark'
import { putControlFile } from '@/lib/sharesync'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

/** Output subdir from the job command: ['bash', spark_start.sh, /output/<sub>, ...]. */
function controlKey(job: { command?: unknown }): string | null {
  const cmd = job.command
  if (!Array.isArray(cmd)) return null
  const arg = cmd.find(a => typeof a === 'string' && a.startsWith('/output/')) as string | undefined
  if (!arg) return null
  return arg.slice('/output/'.length).split('/').filter(Boolean)[0] ?? null
}

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const session = await auth()
  if (!session?.user?.id) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { id } = await ctx.params
  let maxSteps: number
  try {
    const b = await req.json() as { maxSteps?: unknown }
    maxSteps = Math.floor(Number(b.maxSteps))
    if (!Number.isFinite(maxSteps) || maxSteps < 0) throw new Error('bad')
  } catch {
    return NextResponse.json({ error: 'maxSteps must be a non-negative integer (0 = unlimited)' }, { status: 400 })
  }

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
      error: `job is '${job.status}', not running — its max_steps can't be changed. Use Cancel, or set it when creating the job.`,
      notLive: true,
    }, { status: 409 })
  }

  try {
    await putControlFile(key, 'stop_request.json', { nonce: Date.now(), maxSteps })
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : 'failed to write request' }, { status: 502 })
  }
  return NextResponse.json({
    maxSteps,
    note: maxSteps === 0
      ? 'run un-capped (unlimited) — it will keep training until you stop it'
      : 'the running job will stop gracefully at this step (or shortly, if already past it)',
  })
}
