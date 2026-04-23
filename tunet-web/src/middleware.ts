import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? ''
const DEMO_MODE = !clerkKey || !clerkKey.startsWith('pk_') || clerkKey.includes('placeholder')

// Public paths that never require auth
const isPublicRoute = createRouteMatcher([
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/demo(.*)',
  '/api/webhooks/(.*)',
  '/api/billing/webhook',
])

// In demo mode: plain passthrough, no Clerk involved
function demoMiddleware(_req: NextRequest) {
  return NextResponse.next()
}

// In production: full Clerk auth enforcement
const prodMiddleware = clerkMiddleware(async (auth, req: NextRequest) => {
  if (!isPublicRoute(req)) {
    const { userId } = await auth()
    if (!userId) {
      const signInUrl = new URL('/sign-in', req.url)
      signInUrl.searchParams.set('redirect_url', req.url)
      return NextResponse.redirect(signInUrl)
    }
  }
})

export default DEMO_MODE ? demoMiddleware : prodMiddleware

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
}
