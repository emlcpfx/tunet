import { redirect } from 'next/navigation'

const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? ''
const DEMO_MODE = !clerkKey || !clerkKey.startsWith('pk_') || clerkKey.includes('placeholder')

export default async function RootPage() {
  // Demo mode: skip Clerk entirely and land on the live cloud dashboard.
  if (DEMO_MODE) {
    redirect('/demo/dashboard')
  }

  // Production: gated by Clerk
  const { auth } = await import('@clerk/nextjs/server')
  const { userId } = await auth()
  if (userId) redirect('/dashboard')
  else redirect('/sign-in')
}
