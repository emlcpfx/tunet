import NextAuth from 'next-auth'
import Keycloak from 'next-auth/providers/keycloak'
import { createServiceClient } from '@/lib/supabase'

async function syncUserToSupabase(profile: {
  sub?: string | unknown
  email?: string | unknown
  name?: string | unknown
  preferred_username?: string | unknown
  given_name?: string | unknown
  family_name?: string | unknown
}) {
  const sub = typeof profile.sub === 'string' ? profile.sub : null
  if (!sub) return

  const email =
    (typeof profile.email === 'string' && profile.email) ||
    (typeof profile.preferred_username === 'string' && profile.preferred_username) ||
    ''

  let name: string | null =
    typeof profile.name === 'string' && profile.name.trim()
      ? profile.name.trim()
      : null
  if (!name) {
    const gn = typeof profile.given_name === 'string' ? profile.given_name : ''
    const fn = typeof profile.family_name === 'string' ? profile.family_name : ''
    const combined = [gn, fn].filter(Boolean).join(' ')
    name = combined || null
  }

  const svc = createServiceClient()
  const { data: existing } = await svc.from('users').select('id').eq('id', sub).maybeSingle()

  if (!existing) {
    const { error } = await svc.from('users').insert({
      id: sub,
      email,
      name,
      credit_balance_cents: 0,
      is_admin: false,
    })
    if (error) throw error
  } else {
    const { error } = await svc.from('users').update({ email, name }).eq('id', sub)
    if (error) throw error
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  trustHost: true,
  pages: {
    signIn: '/sign-in',
  },
  providers: [
    Keycloak({
      clientId: process.env.AUTH_KEYCLOAK_ID ?? '',
      clientSecret: process.env.AUTH_KEYCLOAK_SECRET ?? '',
      issuer: process.env.AUTH_KEYCLOAK_ISSUER ?? '',
    }),
  ],
  callbacks: {
    async signIn({ profile }) {
      if (!profile?.sub || typeof profile.sub !== 'string') return false
      try {
        await syncUserToSupabase(profile)
      } catch (e) {
        console.error('[auth] Supabase user sync failed', e)
        return false
      }
      return true
    },
    async jwt({ token, profile, account }) {
      if (profile && typeof profile === 'object') {
        const p = profile as Record<string, unknown>
        if (typeof p.sub === 'string') token.sub = p.sub
        if (typeof p.email === 'string') token.email = p.email
        if (typeof p.name === 'string') token.name = p.name
        else if (typeof p.preferred_username === 'string')
          token.name = p.preferred_username as string
      }
      if (account?.providerAccountId && !token.sub) {
        token.sub = account.providerAccountId
      }
      return token
    },
    async session({ session, token }) {
      if (token.sub) session.user.id = token.sub as string
      if (typeof token.email === 'string') session.user.email = token.email
      if (typeof token.name === 'string') session.user.name = token.name
      return session
    },
  },
})
