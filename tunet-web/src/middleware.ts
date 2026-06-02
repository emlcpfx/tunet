import { auth } from '@/auth'
import { NextResponse } from 'next/server'

function isPublicPath(pathname: string): boolean {
  return (
    pathname.startsWith('/api/auth') ||
    pathname.startsWith('/sign-in') ||
    pathname.startsWith('/sign-up') ||
    pathname.startsWith('/api/webhooks') ||
    // Running jobs (no browser session) swap their refresh token for a fresh
    // access token here; it authenticates via the refresh token in the body.
    pathname.startsWith('/api/spark/refresh')
  )
}

export default auth((req) => {
  const pathname = req.nextUrl.pathname
  if (isPublicPath(pathname)) {
    return NextResponse.next()
  }

  // A session is usable only if it has a user id AND its Keycloak token didn't
  // fail to refresh. A failed refresh leaves the *stale* token in place and
  // stamps error='RefreshAccessTokenError' (src/auth.ts) while keeping the user
  // id — so without the error check a long-idle tab keeps acting with a dead
  // bearer: every Spark call 401s and (pre-fix) a cancel could silently no-op
  // while the job kept billing.
  const session = req.auth
  const authed =
    !!session?.user?.id && session.error !== 'RefreshAccessTokenError'
  if (authed) {
    return NextResponse.next()
  }

  // API routes must NEVER be redirected: fetch() transparently follows a 307 to
  // the /sign-in HTML page and hands the caller a 200, so a dead session reads
  // as success in any handler that only checks res.ok. Return a clean 401 JSON
  // (shape matches the cancel route) so the client can surface a re-login prompt.
  if (pathname.startsWith('/api/')) {
    return NextResponse.json(
      { error: 'Your session expired. Sign in again, then retry.', authExpired: true },
      { status: 401 },
    )
  }

  const signIn = new URL('/sign-in', req.url)
  const callback = `${pathname}${req.nextUrl.search}`
  signIn.searchParams.set('callbackUrl', callback || '/')
  return NextResponse.redirect(signIn)
})

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
}
