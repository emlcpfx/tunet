import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { createServiceClient } from '@/lib/supabase'
import { stopPod, terminatePod } from '@/lib/runpod'
import type { DbJob } from '@/types'

type Params = { params: Promise<{ id: string }> }

export async function GET(_req: Request, { params }: Params) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { id } = await params
  const svc = createServiceClient()

  const { data, error } = await svc
    .from('jobs')
    .select('*')
    .eq('id', id)
    .eq('user_id', userId)
    .single()

  if (error || !data) return NextResponse.json({ error: 'Not found' }, { status: 404 })
  return NextResponse.json(data)
}

export async function PATCH(req: Request, { params }: Params) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { id } = await params
  const { action } = await req.json() as { action: 'stop' | 'terminate' }

  const svc = createServiceClient()
  const { data: job } = await svc
    .from('jobs')
    .select('pod_id, status, user_id')
    .eq('id', id)
    .eq('user_id', userId)
    .single()

  if (!job) return NextResponse.json({ error: 'Not found' }, { status: 404 })

  const j = job as Pick<DbJob, 'pod_id' | 'status' | 'user_id'>

  if (!j.pod_id) {
    return NextResponse.json({ error: 'No pod associated with this job' }, { status: 400 })
  }

  try {
    if (action === 'stop') {
      await stopPod(j.pod_id)
      await svc
        .from('jobs')
        .update({ status: 'stopped' })
        .eq('id', id)
      return NextResponse.json({ ok: true, status: 'stopped' })
    }

    if (action === 'terminate') {
      await terminatePod(j.pod_id)
      await svc
        .from('jobs')
        .update({ status: 'terminated', ended_at: new Date().toISOString() })
        .eq('id', id)
      return NextResponse.json({ ok: true, status: 'terminated' })
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 })
  } catch (err) {
    return NextResponse.json({
      error: err instanceof Error ? err.message : 'Action failed',
    }, { status: 500 })
  }
}
