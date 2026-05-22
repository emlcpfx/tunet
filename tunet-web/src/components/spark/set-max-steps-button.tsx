'use client'
import { useState } from 'react'

/**
 * Set/adjust a running training job's stop point (max_steps) live. Writes a
 * control flag the training loop polls; it stops gracefully at that step (saving
 * a final checkpoint), or shortly if already past it. 0 = unlimited (un-cap).
 * Only rendered for live training jobs.
 */
export function SetMaxStepsButton({ jobId }: { jobId: string }) {
  const [open, setOpen] = useState(false)
  const [val, setVal]   = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg]   = useState<string | null>(null)
  const [err, setErr]   = useState<string | null>(null)

  async function apply() {
    const n = Math.floor(Number(val))
    if (!Number.isFinite(n) || n < 0) { setErr('Enter a step number (0 = unlimited)'); return }
    setBusy(true); setErr(null); setMsg(null)
    try {
      const res = await fetch(`/api/spark/jobs/${jobId}/set-max-steps`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ maxSteps: n }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
      setMsg(data.note ?? 'Stop point updated.')
      setOpen(false)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'failed')
    } finally {
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <div className="flex flex-col items-end gap-1">
        <button
          onClick={() => { setOpen(true); setMsg(null) }}
          className="px-3 py-1.5 rounded-md text-xs font-semibold border border-[#e5e7eb] text-[#374151] hover:bg-[#F9FAFB] transition-colors"
          title="Set a graceful stop point (max steps) for this running job"
        >
          Set max steps
        </button>
        {msg && <p className="text-xs text-[#16a34a] max-w-[220px] text-right">{msg}</p>}
      </div>
    )
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-1">
        <input
          type="number" min={0} autoFocus value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void apply(); if (e.key === 'Escape') { setOpen(false); setErr(null) } }}
          placeholder="step (0 = ∞)"
          className="w-28 px-2 py-1.5 text-xs border border-[#e5e7eb] rounded-md focus:border-[#7E3AF2] focus:outline-none"
        />
        <button
          onClick={() => void apply()} disabled={busy}
          className="px-2.5 py-1.5 rounded-md text-xs font-semibold bg-[#7E3AF2] text-white hover:bg-[#6D28D9] disabled:opacity-50"
        >
          {busy ? '…' : 'Set'}
        </button>
        <button
          onClick={() => { setOpen(false); setErr(null) }}
          className="px-2 py-1.5 rounded-md text-xs text-[#6b7280] hover:bg-[#F9FAFB]"
          aria-label="Cancel"
        >
          ✕
        </button>
      </div>
      {err && <p className="text-xs text-[#EF4444] max-w-[220px] text-right">{err}</p>}
    </div>
  )
}
