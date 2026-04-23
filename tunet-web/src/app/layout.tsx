import type { Metadata } from 'next'
import { ClerkProvider } from '@clerk/nextjs'
import './globals.css'

export const metadata: Metadata = {
  title: 'TuNet Cloud',
  description: 'Managed GPU training for TuNet models',
}

const FONTS = (
  <>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
    <link
      href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Fira+Mono:wght@400;500&display=swap"
      rel="stylesheet"
    />
  </>
)

const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Skip ClerkProvider when no real key is set (demo / local UI preview)
  if (!clerkKey || clerkKey.includes('placeholder')) {
    return (
      <html lang="en">
        <head>{FONTS}</head>
        <body>{children}</body>
      </html>
    )
  }

  return (
    <ClerkProvider>
      <html lang="en">
        <head>{FONTS}</head>
        <body>{children}</body>
      </html>
    </ClerkProvider>
  )
}
