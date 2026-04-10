import Stripe from 'stripe'

// Lazy singleton — avoids build-time failure when env var isn't set
let _stripe: Stripe | null = null
export function getStripe(): Stripe {
  if (!_stripe) {
    if (!process.env.STRIPE_SECRET_KEY) throw new Error('STRIPE_SECRET_KEY not set')
    _stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: '2025-02-24.acacia' })
  }
  return _stripe
}


export interface CreateCheckoutSessionParams {
  userId: string
  userEmail: string
  balanceCents: number  // amount added to user's balance (cents)
  priceCents: number    // what Stripe charges (cents, slightly higher to cover fees)
  packLabel: string
}

export async function createCheckoutSession(
  params: CreateCheckoutSessionParams,
): Promise<string> {
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'

  const session = await getStripe().checkout.sessions.create({
    mode: 'payment',
    payment_method_types: ['card'],
    customer_email: params.userEmail,
    line_items: [
      {
        quantity: 1,
        price_data: {
          currency: 'usd',
          unit_amount: params.priceCents,
          product_data: {
            name: `TuNet Cloud — ${params.packLabel}`,
            description: `$${(params.balanceCents / 100).toFixed(2)} added to your compute balance`,
          },
        },
      },
    ],
    metadata: {
      userId:       params.userId,
      balanceCents: String(params.balanceCents),
    },
    success_url: `${appUrl}/billing?success=1`,
    cancel_url:  `${appUrl}/billing?cancelled=1`,
  })

  return session.url!
}
