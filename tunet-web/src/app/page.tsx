import { redirect } from 'next/navigation'
import { auth } from '@/auth'

export default async function RootPage() {
  const session = await auth()
  if (session?.user?.id) redirect('/dashboard')
  redirect('/sign-in')
}
