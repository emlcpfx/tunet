import { NextResponse } from 'next/server'
import { runBillingTick } from '@/lib/billing'

// Protected by Vercel's cron secret — injected automatically
export async function GET(req: Request) {
  const authHeader = req.headers.get('authorization')
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    const result = await runBillingTick()
    return NextResponse.json({
      ok: true,
      charged_cents: result.charged,
      jobs_stopped:  result.stopped,
      timestamp:     new Date().toISOString(),
    })
  } catch (err) {
    console.error('[billing-tick] Error:', err)
    return NextResponse.json({
      error: err instanceof Error ? err.message : 'Billing tick failed',
    }, { status: 500 })
  }
}
