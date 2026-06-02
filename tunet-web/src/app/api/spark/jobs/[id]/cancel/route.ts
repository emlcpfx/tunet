/**
 * POST /api/spark/jobs/:id/cancel — send SIGTERM to the running container
 */

import { NextResponse } from 'next/server'
import { cancelJob, SparkAuthError } from '@/lib/spark'

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
    // Expired/invalid session → 401 + authExpired so the client prompts a
    // re-login instead of swallowing it. This is the case that let a cancel
    // silently no-op on a days-old tab while the job kept billing.
    if (e instanceof SparkAuthError) {
      return NextResponse.json(
        { error: e.message, authExpired: true },
        { status: 401 },
      )
    }
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Cancel failed' },
      { status: 500 },
    )
  }
}
