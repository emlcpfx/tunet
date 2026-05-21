/**
 * POST /api/spark/jobs/:id/request-export   body: { flame?, nuke? }
 *
 * Asks a RUNNING training job to export the current model inline. Drops an
 * `export_request.json` flag in the job's ShareSync control path; train.py polls
 * it once per epoch and exports + self-uploads on a new nonce. No new compute —
 * the model is already on the training GPU.
 *
 * Only meaningful while the job is running; if it's already finished, the model
 * isn't loaded anywhere, so the caller should fall back to the on-demand export
 * job (or just use the final auto-export the run already produced).
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
  let flame = true, nuke = false
  try {
    const b = await req.json() as { flame?: boolean; nuke?: boolean }
    if (typeof b.flame === 'boolean') flame = b.flame
    if (typeof b.nuke === 'boolean')  nuke  = b.nuke
  } catch { /* defaults: flame only */ }
  if (!flame && !nuke) return NextResponse.json({ error: 'request at least one of flame/nuke' }, { status: 400 })

  let job
  try { job = await getJob(id) } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : 'failed' }, { status: 502 })
  }
  if (!job) return NextResponse.json({ error: 'job not found' }, { status: 404 })

  const key = controlKey(job)
  if (!key) return NextResponse.json({ error: 'cannot derive output subdir for this job' }, { status: 422 })

  // Running-ish only — a finished job can't export inline (model not loaded).
  const status = String(job.status ?? '').toLowerCase()
  const live = !['succeeded', 'completed', 'failed', 'cancelled', 'stopped', 'error'].includes(status)
  if (!live) {
    return NextResponse.json({
      error: `job is '${job.status}', not running — its model isn't loaded, so an inline export isn't possible. Use the final/auto export or the on-demand export job.`,
      notLive: true,
    }, { status: 409 })
  }

  try {
    await putControlFile(key, 'export_request.json', { nonce: Date.now(), flame, nuke })
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : 'failed to write request' }, { status: 502 })
  }
  return NextResponse.json({ requested: { flame, nuke }, note: 'the running job will export at its next epoch boundary' })
}
