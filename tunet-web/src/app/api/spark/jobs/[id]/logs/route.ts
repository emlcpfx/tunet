/**
 * GET /api/spark/jobs/:id/logs
 *
 * Server-Sent Events proxy. Opens the upstream Spark SSE stream with the
 * server-side Bearer token and pipes the body straight to the browser. The
 * client can connect with EventSource('/api/spark/jobs/...').
 */

import { openLogStream } from '@/lib/spark'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'   // need streaming response — Edge would also work

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params
  let upstream: Response
  try {
    upstream = await openLogStream(id)
  } catch (e) {
    return new Response(
      `event: error\ndata: ${e instanceof Error ? e.message : 'failed'}\n\n`,
      {
        status: 500,
        headers: { 'Content-Type': 'text/event-stream' },
      },
    )
  }

  if (!upstream.ok || !upstream.body) {
    return new Response(
      `event: error\ndata: upstream HTTP ${upstream.status}\n\n`,
      {
        status: upstream.status,
        headers: { 'Content-Type': 'text/event-stream' },
      },
    )
  }

  // Pipe upstream → client. Forward SSE-relevant headers; don't compress.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type':       'text/event-stream',
      'Cache-Control':      'no-cache, no-transform',
      'Connection':         'keep-alive',
      'X-Accel-Buffering':  'no',   // Vercel/Nginx hint to disable buffering
    },
  })
}
