'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'

export function CancelJobButton({ jobId }: { jobId: string }) {
  const router = useRouter()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [authExpired, setAuthExpired] = useState(false)

  async function onCancel() {
    if (!window.confirm('Send SIGTERM to this job?\n\nWorst case you lose ≤500 steps since the last save.')) {
      return
    }
    setBusy(true)
    setError(null)
    setAuthExpired(false)
    try {
      const res = await fetch(`/api/spark/jobs/${jobId}/cancel`, { method: 'POST' })
      if (!res.ok) {
        const j = await res.json().catch(() => ({}))
        if (j.authExpired) setAuthExpired(true)
        throw new Error(j.error ?? `Cancel failed (HTTP ${res.status})`)
      }
      // Refresh server-rendered chrome (status badge etc.) to reflect cancelled state
      router.refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Cancel failed')
    } finally {
      setBusy(false)
    }
  }

  const signInHref =
    typeof window !== 'undefined'
      ? `/sign-in?callbackUrl=${encodeURIComponent(window.location.pathname)}`
      : '/sign-in'

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={onCancel}
        disabled={busy}
        className="px-3 py-1.5 rounded-md text-xs font-semibold bg-[#FEF2F2] text-[#EF4444] border border-[#fecaca] hover:bg-[#FEE2E2] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {busy ? 'Cancelling…' : 'Cancel Job'}
      </button>
      {error && (
        <p className="text-xs text-[#EF4444] text-right max-w-[260px]">
          {error}
          {authExpired && (
            <>
              {' '}
              <a href={signInHref} className="underline font-semibold">
                Sign in again
              </a>
            </>
          )}
        </p>
      )}
    </div>
  )
}
