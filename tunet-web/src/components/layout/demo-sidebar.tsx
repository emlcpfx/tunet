'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { signOut } from 'next-auth/react'
import { SparkLogo } from '@/components/brand/spark-logo'

const navItems = [
  {
    href: '/demo/dashboard',
    label: 'Dashboard',
    exact: true,
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
    ),
  },
  {
    href: '/demo/comfy',
    label: 'EZ-Comfy',
    exact: false,
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/>
      </svg>
    ),
  },
  {
    href: '/demo/billing',
    label: 'Billing',
    exact: false,
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/>
      </svg>
    ),
  },
]

interface DemoSidebarProps {
  userName?: string
  authenticated?: boolean
}

export function DemoSidebar({ userName = 'Demo User', authenticated = false }: DemoSidebarProps) {
  const pathname = usePathname()

  function isActive(item: { href: string; exact: boolean }) {
    if (item.exact) return pathname === item.href
    return pathname.startsWith(item.href)
  }

  return (
    <nav className="sidebar">
      <div className="flex items-center gap-2.5 px-4 py-5 border-b border-[#e5e7eb]">
        <SparkLogo />
        <span className="text-lg font-bold text-[#111827]">Spark</span>
        <span className="text-xs text-[#9ca3af] mt-0.5">Flint</span>
      </div>

      <div className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const active = isActive(item)
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors duration-150 ${
                active
                  ? 'text-[#ae69f4] bg-[#F7F4FC] border-l-2 border-[#ae69f4] pl-[10px]'
                  : 'text-[#374151] hover:bg-[#F9FAFB]'
              }`}
            >
              <span className={active ? 'text-[#ae69f4]' : 'text-[#6b7280]'}>{item.icon}</span>
              {item.label}
            </Link>
          )
        })}
      </div>

      <div className="border-t border-[#e5e7eb] px-3 py-3 space-y-1">
        <div className="flex items-center gap-2.5 px-2 py-1.5">
          <div className="w-7 h-7 rounded-full bg-[#ae69f4] flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
            {userName[0]?.toUpperCase() ?? 'U'}
          </div>
          <span className="text-sm text-[#374151] truncate flex-1">{userName}</span>
        </div>
        {authenticated ? (
          <button
            type="button"
            onClick={() => void signOut({ callbackUrl: '/sign-in' })}
            className="w-full text-left px-2 py-1.5 text-xs text-[#6b7280] hover:text-[#374151] hover:bg-[#F9FAFB] rounded-md transition-colors"
          >
            Log out
          </button>
        ) : (
          <div className="px-2 py-1 mt-1">
            <span className="text-xs text-[#9ca3af] italic">Demo mode</span>
          </div>
        )}
      </div>
    </nav>
  )
}
