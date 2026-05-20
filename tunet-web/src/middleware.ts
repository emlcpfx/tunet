import { auth } from '@/auth'
import { NextResponse } from 'next/server'

function isPublicPath(pathname: string): boolean {
  return (
    pathname.startsWith('/api/auth') ||
    pathname.startsWith('/sign-in') ||
    pathname.startsWith('/sign-up') ||
    pathname.startsWith('/api/webhooks')
  )
}

export default auth((req) => {
  const pathname = req.nextUrl.pathname
  if (isPublicPath(pathname)) {
    return NextResponse.next()
  }

  const userId = req.auth?.user?.id
  if (!userId) {
    const signIn = new URL('/sign-in', req.url)
    const callback = `${pathname}${req.nextUrl.search}`
    signIn.searchParams.set('callbackUrl', callback || '/dashboard')
    return NextResponse.redirect(signIn)
  }

  return NextResponse.next()
})

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
}
