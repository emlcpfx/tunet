/**
 * POST /api/spark/jobs/:id/cancel — send SIGTERM to the running container
 */

import { NextResponse } from 'next/server'
import { cancelJob } from '@/lib/spark'

export const dynamic = 'force-dynamic'

export async function POST(
  _req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params
  try {
    const job = await cancelJob(id)
    return NextResponse.json({ job })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Cancel failed' },
      { status: 500 },
    )
  }
}
