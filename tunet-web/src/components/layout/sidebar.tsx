'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useUser, useClerk } from '@clerk/nextjs'
import { formatCredits } from '@/types'

interface NavItem {
  href: string
  label: string
  icon: React.ReactNode
  exact?: boolean
}

function SparkLogo() {
  return (
    <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-8 h-8">
      <circle cx="16" cy="16" r="14" fill="#ae69f4"/>
      <path d="M12 10c0-1 1.5-2 4-2s4 1 4 2c0 2-4 3-4 5 0 1.5 0 2 0 2m0 3v1"
        stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  )
}

const navItems: NavItem[] = [
  {
    href: '/dashboard',
    label: 'Dashboard',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
    ),
    exact: true,
  },
  {
    href: '/jobs',
    label: 'Training Jobs',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a4 4 0 0 0-4 4c0 2 1 3 2 4l-3 3a2 2 0 0 0 0 3h10a2 2 0 0 0 0-3l-3-3c1-1 2-2 2-4a4 4 0 0 0-4-4z"/>
        <circle cx="12" cy="19" r="1.5"/>
      </svg>
    ),
  },
  {
    href: '/billing',
    label: 'Billing',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect width="20" height="14" x="2" y="5" rx="2"/>
        <line x1="2" x2="22" y1="10" y2="10"/>
      </svg>
    ),
  },
  {
    href: '/settings',
    label: 'Settings',
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
      </svg>
    ),
  },
]

interface SidebarProps {
  creditBalance?: number
}

export function Sidebar({ creditBalance }: SidebarProps) {
  const pathname = usePathname()
  const { user } = useUser()
  const { signOut } = useClerk()

  function isActive(item: NavItem) {
    if (item.exact) return pathname === item.href
    return pathname.startsWith(item.href)
  }

  return (
    <nav className="sidebar">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5 border-b border-[#e5e7eb]">
        <SparkLogo />
        <span className="text-lg font-bold text-[#111827]">TuNet</span>
        <span className="text-xs text-[#9ca3af] mt-0.5">Cloud</span>
      </div>

      {/* Nav */}
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

      {/* Credit balance */}
      {creditBalance !== undefined && (
        <div className="mx-3 mb-2 px-3 py-2.5 bg-[#F7F4FC] rounded-lg">
          <p className="text-xs text-[#6b7280]">Credits</p>
          <p className="text-base font-bold text-[#ae69f4]">{formatCredits(creditBalance)}</p>
        </div>
      )}

      {/* User */}
      <div className="border-t border-[#e5e7eb] px-3 py-3 space-y-1">
        <div className="flex items-center gap-2.5 px-2 py-1.5">
          <div className="w-7 h-7 rounded-full bg-[#ae69f4] flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
            {user?.firstName?.[0] ?? user?.emailAddresses[0]?.emailAddress[0]?.toUpperCase() ?? '?'}
          </div>
          <span className="text-sm text-[#374151] truncate flex-1">
            {user?.firstName ?? user?.emailAddresses[0]?.emailAddress ?? 'User'}
          </span>
        </div>
        <button
          onClick={() => signOut({ redirectUrl: '/sign-in' })}
          className="w-full text-left px-2 py-1.5 text-xs text-[#6b7280] hover:text-[#374151] hover:bg-[#F9FAFB] rounded-md transition-colors"
        >
          Log out
        </button>
      </div>
    </nav>
  )
}
