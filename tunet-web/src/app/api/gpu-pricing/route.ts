import { NextResponse } from 'next/server'
import { createServiceClient } from '@/lib/supabase'
import { DEFAULT_GPU_PRICING } from '@/lib/pricing'

export async function GET() {
  const svc = createServiceClient()
  const { data, error } = await svc
    .from('gpu_pricing')
    .select('*')
    .eq('is_available', true)
    .order('sort_order')

  // If table is empty (fresh setup), return defaults
  if (error || !data || data.length === 0) {
    return NextResponse.json(DEFAULT_GPU_PRICING.filter(g => g.is_available))
  }

  return NextResponse.json(data)
}
