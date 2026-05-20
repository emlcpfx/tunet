import NextAuth from 'next-auth'
import Keycloak from 'next-auth/providers/keycloak'
import type { JWT } from 'next-auth/jwt'

const issuer = process.env.AUTH_KEYCLOAK_ISSUER ?? ''
const clientId = process.env.AUTH_KEYCLOAK_ID ?? ''
const clientSecret = process.env.AUTH_KEYCLOAK_SECRET ?? ''

// Refresh skew: renew the access token a minute before it actually expires so
// an in-flight Spark call never rides an already-dead token.
const REFRESH_SKEW_MS = 60_000

/**
 * Exchange the stored refresh token for a fresh access token at Keycloak's
 * token endpoint. Keycloak rotates refresh tokens, so we persist whichever it
 * returns. On failure we flag the token so the session can bounce to sign-in.
 */
async function refreshAccessToken(token: JWT): Promise<JWT> {
  try {
    if (!token.refreshToken) throw new Error('no refresh token')
    const res = await fetch(`${issuer}/protocol/openid-connect/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        client_id: clientId,
        client_secret: clientSecret,
        refresh_token: token.refreshToken,
      }),
      cache: 'no-store',
    })
    const data = (await res.json()) as {
      access_token?: string
      refresh_token?: string
      expires_in?: number
      error?: string
    }
    if (!res.ok || !data.access_token) {
      throw new Error(data.error ?? `HTTP ${res.status}`)
    }
    return {
      ...token,
      accessToken: data.access_token,
      accessTokenExpires: Date.now() + (data.expires_in ?? 300) * 1000,
      refreshToken: data.refresh_token ?? token.refreshToken,
      error: undefined,
    }
  } catch (e) {
    console.error('[auth] token refresh failed', e)
    return { ...token, error: 'RefreshAccessTokenError' }
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  trustHost: true,
  pages: {
    signIn: '/sign-in',
  },
  providers: [
    Keycloak({ clientId, clientSecret, issuer }),
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      // Initial sign-in: account is present. Persist identity + the Keycloak
      // tokens we forward to Spark. Identity comes from Keycloak/Spark's realm
      // (the single source of truth) — we no longer mirror users into Supabase.
      if (account) {
        if (typeof profile?.sub === 'string') token.sub = profile.sub
        if (typeof profile?.email === 'string') token.email = profile.email
        if (typeof profile?.name === 'string') token.name = profile.name
        else if (typeof profile?.preferred_username === 'string')
          token.name = profile.preferred_username as string
        token.accessToken = account.access_token
        token.refreshToken = account.refresh_token
        token.accessTokenExpires = account.expires_at
          ? account.expires_at * 1000
          : Date.now() + 300_000
        token.error = undefined
        return token
      }

      // Still valid → use as-is.
      if (
        token.accessTokenExpires &&
        Date.now() < token.accessTokenExpires - REFRESH_SKEW_MS
      ) {
        return token
      }

      // Expired (or no expiry recorded) → try to refresh.
      return refreshAccessToken(token)
    },
    async session({ session, token }) {
      if (token.sub) session.user.id = token.sub
      if (typeof token.email === 'string') session.user.email = token.email
      if (typeof token.name === 'string') session.user.name = token.name
      // accessToken is deliberately omitted — it stays in the encrypted JWT and
      // is read server-side only, never sent to the browser.
      if (token.error) session.error = token.error
      return session
    },
  },
})
