import { auth } from '@clerk/nextjs/server'
import { redirect } from 'next/navigation'
import { createServiceClient } from '@/lib/supabase'
import { Sidebar } from '@/components/layout/sidebar'

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { userId } = await auth()
  if (!userId) redirect('/sign-in')

  // Fetch credit balance
  const svc = createServiceClient()
  const { data: user } = await svc
    .from('users')
    .select('credit_balance_cents')
    .eq('id', userId)
    .single()

  return (
    <div className="flex min-h-screen">
      <Sidebar creditBalance={user?.credit_balance_cents ?? 0} />
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-6xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
