'use client'

import { Suspense, useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { signIn } from 'next-auth/react'
import { SparkLogo } from '@/components/brand/spark-logo'

function SignInForm() {
  const searchParams = useSearchParams()

  const raw = searchParams.get('callbackUrl') ?? ''
  const safe =
    raw.startsWith('/') && !raw.startsWith('//') ? raw : '/dashboard'

  // The middleware appends a callbackUrl when it bounces an unauthenticated
  // user off a protected page; signOut sends them here without one. So treat
  // "callbackUrl present (and no auth error)" as the signal to auto-redirect
  // to Keycloak — a signed-out user keeps the manual button instead of being
  // silently re-authenticated through Keycloak's SSO session.
  const autoRedirect = raw !== '' && !searchParams.get('error')
  const [busy, setBusy] = useState(autoRedirect)

  async function onContinue() {
    setBusy(true)
    await signIn('keycloak', { callbackUrl: safe })
  }

  useEffect(() => {
    if (autoRedirect) void signIn('keycloak', { callbackUrl: safe })
  }, [autoRedirect, safe])

  return (
    <>
      <button
        type="button"
        disabled={busy}
        onClick={() => void onContinue()}
        className="flex w-full justify-center rounded-xl bg-[#ae69f4] px-4 py-3 text-sm font-semibold text-white hover:bg-[#7E3AF2] transition-colors disabled:opacity-60"
      >
        {busy ? 'Redirecting…' : 'Continue with Keycloak'}
      </button>
      <p className="mt-6 text-center text-xs text-[#9ca3af]">
        Self-registration is configured in your Keycloak realm. Use the same button after your admin creates your account.
      </p>
    </>
  )
}

export default function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F9FAFB]">
      <div className="w-full max-w-md px-4">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2.5 mb-4">
            <SparkLogo className="w-10 h-10" />
            <span className="text-2xl font-bold text-[#111827]">Spark Flint</span>
          </div>
          <p className="text-sm text-[#6b7280]">Sign in with your organization account</p>
        </div>
        <Suspense
          fallback={
            <div className="h-11 rounded-xl bg-[#e5e7eb] animate-pulse" aria-hidden />
          }
        >
          <SignInForm />
        </Suspense>
      </div>
    </div>
  )
}
