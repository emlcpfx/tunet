import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { createServiceClient } from '@/lib/supabase'

export async function GET() {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const svc = createServiceClient()
  const { data } = await svc
    .from('users')
    .select('credit_balance_cents')
    .eq('id', userId)
    .single()

  return NextResponse.json({ balance_cents: data?.credit_balance_cents ?? 0 })
}
