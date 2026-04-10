export const dynamic = 'force-dynamic'

import { NextResponse } from 'next/server'
import { headers } from 'next/headers'
import { getStripe } from '@/lib/stripe'
import { createServiceClient } from '@/lib/supabase'
import type Stripe from 'stripe'

export async function POST(req: Request) {
  const body = await req.text()
  const headersList = await headers()
  const sig = headersList.get('stripe-signature')

  if (!sig) return NextResponse.json({ error: 'No signature' }, { status: 400 })

  let event: Stripe.Event
  try {
    event = getStripe().webhooks.constructEvent(
      body,
      sig,
      process.env.STRIPE_WEBHOOK_SECRET!,
    )
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Webhook verification failed' },
      { status: 400 },
    )
  }

  if (event.type === 'checkout.session.completed') {
    const session = event.data.object as Stripe.Checkout.Session
    const userId  = session.metadata?.userId
    const credits = parseInt(session.metadata?.balanceCents ?? session.metadata?.credits ?? '0', 10)

    if (!userId || !credits) {
      return NextResponse.json({ error: 'Missing metadata' }, { status: 400 })
    }

    const svc = createServiceClient()

    // Idempotency: check if we've processed this payment intent already
    const paymentIntent = typeof session.payment_intent === 'string'
      ? session.payment_intent
      : session.payment_intent?.id

    if (paymentIntent) {
      const { data: existing } = await svc
        .from('billing_events')
        .select('id')
        .eq('stripe_payment_intent', paymentIntent)
        .single()

      if (existing) {
        return NextResponse.json({ ok: true, idempotent: true })
      }
    }

    // Add credits to user balance
    const { error: updateErr } = await svc.rpc('add_credits', {
      p_user_id:     userId,
      p_amount_cents: credits,
    })

    if (updateErr) {
      console.error('Failed to add credits:', updateErr)
      return NextResponse.json({ error: updateErr.message }, { status: 500 })
    }

    // Record billing event
    await svc.from('billing_events').insert({
      user_id:                userId,
      type:                   'top_up',
      amount_cents:           credits,
      description:            `Credit top-up via Stripe`,
      stripe_payment_intent:  paymentIntent,
    })
  }

  return NextResponse.json({ received: true })
}
