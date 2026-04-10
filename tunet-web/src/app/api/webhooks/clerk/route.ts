import { NextResponse } from 'next/server'
import { Webhook } from 'svix'
import { headers } from 'next/headers'
import { createServiceClient } from '@/lib/supabase'

interface ClerkUserCreatedEvent {
  type: 'user.created' | 'user.updated'
  data: {
    id: string
    email_addresses: Array<{ email_address: string; id: string }>
    first_name: string | null
    last_name: string | null
    primary_email_address_id: string
  }
}

export async function POST(req: Request) {
  const body = await req.text()
  const headersList = await headers()

  const svixId        = headersList.get('svix-id') ?? ''
  const svixTimestamp = headersList.get('svix-timestamp') ?? ''
  const svixSignature = headersList.get('svix-signature') ?? ''

  const wh = new Webhook(process.env.CLERK_WEBHOOK_SECRET!)

  let event: ClerkUserCreatedEvent
  try {
    event = wh.verify(body, {
      'svix-id':        svixId,
      'svix-timestamp': svixTimestamp,
      'svix-signature': svixSignature,
    }) as ClerkUserCreatedEvent
  } catch {
    return NextResponse.json({ error: 'Webhook verification failed' }, { status: 400 })
  }

  if (event.type !== 'user.created' && event.type !== 'user.updated') {
    return NextResponse.json({ ignored: true })
  }

  const { id, email_addresses, first_name, last_name, primary_email_address_id } = event.data
  const primaryEmail = email_addresses.find(e => e.id === primary_email_address_id)?.email_address
    ?? email_addresses[0]?.email_address
    ?? ''

  const name = [first_name, last_name].filter(Boolean).join(' ') || null

  const svc = createServiceClient()

  if (event.type === 'user.created') {
    await svc.from('users').upsert({
      id,
      email: primaryEmail,
      name,
      credit_balance_cents: 0,
      is_admin: false,
    })
  } else {
    await svc
      .from('users')
      .update({ email: primaryEmail, name })
      .eq('id', id)
  }

  return NextResponse.json({ ok: true })
}
