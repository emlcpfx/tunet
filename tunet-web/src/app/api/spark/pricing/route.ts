/**
 * GET /api/spark/pricing — live per-SKU pricing from Spark's estimate endpoint.
 *
 * Calls POST /api/compute/jobs/estimate (Spark Fuse v1.1+) for each SKU in the
 * job-form's GPU picker, in both `instant` and `smart` modes, and returns the
 * billed-per-hour rate that's actually charged to the customer.
 *
 * Server-side 5-minute in-memory cache so a form re-render doesn't fan out to
 * 10 Spark calls every keystroke.
 */

import { NextResponse } from 'next/server'
import { estimateJobCost } from '@/lib/spark'

export const dynamic = 'force-dynamic'

export interface PricingEntry {
  instantUsdPerHr: number | null
  smartUsdPerHr:   number | null
}
export type PricingMap = Record<string, PricingEntry>

const CACHE_TTL_MS = 5 * 60 * 1000

// SKUs the job-form's GPU picker shows. Mirrors GPU_OPTIONS in
// src/app/demo/jobs/new/page.tsx — kept in sync by hand because the form's
// list also encodes display order, badges, and VRAM.
const SKUS = [
  'g4dn.xlarge',   // T4
  'g5.xlarge',     // A10
  'g6.2xlarge',    // L4
  'g6e.8xlarge',   // L40S
  'g7e.2xlarge',   // RTX PRO 6000
] as const

let _cache: { at: number; data: PricingMap } | null = null

async function fetchRate(sku: string, mode: 'instant' | 'smart'): Promise<number | null> {
  try {
    const r = await estimateJobCost(sku, mode)
    const n = parseFloat(r.rate?.billedPerHourUsd ?? '')
    return Number.isFinite(n) ? n : null
  } catch {
    return null
  }
}

export async function GET() {
  if (_cache && Date.now() - _cache.at < CACHE_TTL_MS) {
    return NextResponse.json(_cache.data, { headers: { 'X-Cache': 'HIT' } })
  }

  const entries = await Promise.all(
    SKUS.map(async (sku) => {
      const [instant, smart] = await Promise.all([
        fetchRate(sku, 'instant'),
        fetchRate(sku, 'smart'),
      ])
      return [sku, { instantUsdPerHr: instant, smartUsdPerHr: smart }] as const
    }),
  )

  const data: PricingMap = Object.fromEntries(entries)
  _cache = { at: Date.now(), data }
  return NextResponse.json(data, { headers: { 'X-Cache': 'MISS' } })
}
