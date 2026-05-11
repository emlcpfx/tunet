export const dynamic = 'force-dynamic'

import { auth } from '@/auth'
import { NextResponse } from 'next/server'
import { createCheckoutSession } from '@/lib/stripe'
import { CREDIT_PACKS } from '@/types'

export async function POST(req: Request) {
  const session = await auth()
  const userId = session?.user?.id
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { packId, priceCents, balanceCents } = await req.json() as {
    packId: string
    priceCents: number
    balanceCents: number
  }

  // Validate against known packs
  const pack = CREDIT_PACKS.find(p => p.id === packId)
  if (!pack || pack.price_cents !== priceCents || pack.balance_cents !== balanceCents) {
    return NextResponse.json({ error: 'Invalid pack' }, { status: 400 })
  }

  const email = session.user?.email ?? ''

  const url = await createCheckoutSession({
    userId,
    userEmail: email,
    balanceCents,
    priceCents,
    packLabel: pack.label,
  })

  return NextResponse.json({ url })
}
