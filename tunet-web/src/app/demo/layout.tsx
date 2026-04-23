import { DemoSidebar } from '@/components/layout/demo-sidebar'

export default function DemoLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen">
      <DemoSidebar creditBalance={5250} userName="Eric Levy" />
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-6xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}
