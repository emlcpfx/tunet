import { redirect } from 'next/navigation'
import { auth } from '@/auth'

const keycloakId = process.env.AUTH_KEYCLOAK_ID ?? ''
const DEMO_MODE = !keycloakId || keycloakId === 'your-nextjs-client-id'

export default async function RootPage() {
  if (DEMO_MODE) {
    redirect('/demo/dashboard')
  }

  const session = await auth()
  if (session?.user?.id) redirect('/demo/dashboard')
  redirect('/sign-in')
}
