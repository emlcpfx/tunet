/**
 * POST /api/spark/refresh   body: { refreshToken }
 *
 * Token-refresh proxy for long-running Spark jobs. A submitted job carries the
 * user's Keycloak refresh token (not the short-lived access token); when its
 * in-container self-upload code needs a fresh Spark bearer, it POSTs the refresh
 * token here and we exchange it at Keycloak using the confidential client's
 * secret — which therefore never leaves the server. Returns a fresh access token
 * (and the possibly-rotated refresh token, so the job keeps refreshing).
 *
 * Public (allowlisted in middleware): a running job has no browser session. It
 * authenticates by possessing a valid refresh token, which Keycloak validates —
 * this endpoint is a thin proxy and grants nothing the refresh token didn't
 * already represent.
 */
import { NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const issuer       = process.env.AUTH_KEYCLOAK_ISSUER ?? ''
const clientId     = process.env.AUTH_KEYCLOAK_ID ?? ''
const clientSecret = process.env.AUTH_KEYCLOAK_SECRET ?? ''

export async function POST(req: Request) {
  let refreshToken: string | undefined
  try {
    const body = (await req.json()) as { refreshToken?: string }
    refreshToken = body.refreshToken
  } catch { /* fall through to 400 */ }
  if (!refreshToken) {
    return NextResponse.json({ error: 'refreshToken required' }, { status: 400 })
  }
  if (!issuer || !clientId || !clientSecret) {
    return NextResponse.json({ error: 'Keycloak not configured' }, { status: 503 })
  }

  let res: Response
  try {
    res = await fetch(`${issuer}/protocol/openid-connect/token`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body:    new URLSearchParams({
        grant_type:    'refresh_token',
        client_id:     clientId,
        client_secret: clientSecret,
        refresh_token: refreshToken,
      }),
      cache: 'no-store',
    })
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : 'token endpoint unreachable' }, { status: 502 })
  }

  const data = (await res.json().catch(() => ({}))) as {
    access_token?: string; refresh_token?: string; expires_in?: number; error?: string
  }
  if (!res.ok || !data.access_token) {
    // Don't echo Keycloak internals; a 401 means the refresh token is dead.
    return NextResponse.json({ error: data.error ?? 'refresh failed' }, { status: 401 })
  }
  return NextResponse.json({
    accessToken:  data.access_token,
    expiresIn:    data.expires_in ?? 300,
    refreshToken: data.refresh_token ?? refreshToken,
  })
}
