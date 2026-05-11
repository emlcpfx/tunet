import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { createServiceClient } from '@/lib/supabase'
import { proxyMonitor, monitorBaseUrl } from '@/lib/runpod'
import type { DbJob } from '@/types'

type Params = { params: Promise<{ id: string }> }

// Proxy monitor_api.py endpoints through the server to:
//   1. Enforce job ownership (users can only poll their own jobs)
//   2. Avoid CORS issues
//   3. Keep pod IDs off the client

export async function GET(req: Request, { params }: Params) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { id } = await params
  const url = new URL(req.url)
  const path = url.searchParams.get('path') ?? '/health'

  // Verify ownership
  const svc = createServiceClient()
  const { data: job } = await svc
    .from('jobs')
    .select('pod_id, status')
    .eq('id', id)
    .eq('user_id', userId)
    .single()

  if (!job) return NextResponse.json({ error: 'Not found' }, { status: 404 })

  const j = job as Pick<DbJob, 'pod_id' | 'status'>

  if (!j.pod_id) {
    return NextResponse.json({ data: null, error: 'Pod not yet assigned' }, { status: 200 })
  }

  if (j.status !== 'running' && j.status !== 'provisioning') {
    return NextResponse.json({ data: null, error: 'Pod is not running' }, { status: 200 })
  }

  // Special handling for preview image — stream binary
  if (path === '/api/preview' || path === '/api/val_preview') {
    const imgUrl = `${monitorBaseUrl(j.pod_id)}${path}`
    try {
      const res = await fetch(imgUrl, {
        signal: AbortSignal.timeout(8000),
        next: { revalidate: 0 },
      })
      if (!res.ok) return NextResponse.json({ error: 'Preview not available' }, { status: 404 })
      const buf = await res.arrayBuffer()
      return new NextResponse(buf, {
        headers: {
          'Content-Type': 'image/jpeg',
          'Cache-Control': 'no-cache',
        },
      })
    } catch {
      return NextResponse.json({ error: 'Preview unavailable' }, { status: 404 })
    }
  }

  // JSON endpoints
  const { data, error } = await proxyMonitor(j.pod_id, path)
  return NextResponse.json({ data, error })
}

// POST proxy for /api/stop
export async function POST(req: Request, { params }: Params) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { id } = await params
  const url = new URL(req.url)
  const path = url.searchParams.get('path') ?? '/api/stop'

  const svc = createServiceClient()
  const { data: job } = await svc
    .from('jobs')
    .select('pod_id, status')
    .eq('id', id)
    .eq('user_id', userId)
    .single()

  if (!job) return NextResponse.json({ error: 'Not found' }, { status: 404 })

  const j = job as Pick<DbJob, 'pod_id' | 'status'>
  if (!j.pod_id) return NextResponse.json({ error: 'No pod' }, { status: 400 })

  const { data, error } = await proxyMonitor(j.pod_id, path, { method: 'POST' })
  return NextResponse.json({ data, error })
}
