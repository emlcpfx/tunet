/**
 * GET /api/spark/skus — eligible compute SKUs (server proxy to Spark API)
 */

import { NextResponse } from 'next/server'
import { listSkus, GPU_TYPES } from '@/lib/spark'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const skus = await listSkus()
    return NextResponse.json({ skus, shortcuts: GPU_TYPES })
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'Failed' },
      { status: 500 },
    )
  }
}
