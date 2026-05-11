import type { Metadata } from 'next'
import './globals.css'
import { AuthSessionProvider } from '@/components/providers/session-provider'

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

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>{FONTS}</head>
      <body>
        <AuthSessionProvider>{children}</AuthSessionProvider>
      </body>
    </html>
  )
}
