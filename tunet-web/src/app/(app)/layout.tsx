import { auth } from '@/auth'
import { DemoSidebar } from '@/components/layout/demo-sidebar'

export default async function DemoLayout({ children }: { children: React.ReactNode }) {
  const session = await auth()
  const userId = session?.user?.id

  // Authenticated (production / Keycloak) users get their real identity.
  // Without a session we're in demo mode, so fall back to a generic
  // placeholder rather than leaking a real person's name into screenshots.
  const authenticated = Boolean(userId)
  const userName = userId
    ? (session?.user?.name ?? session?.user?.email ?? 'User')
    : 'Demo User'

  return (
    <div className="flex min-h-screen">
      <DemoSidebar userName={userName} authenticated={authenticated} />
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-6xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
