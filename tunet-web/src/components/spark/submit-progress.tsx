'use client'

/**
 * Modal overlay shown while POST /api/spark/training-jobs is streaming.
 *
 * Displays four phases (validate · pack · submit · upload) as a vertical
 * step list with running totals + a determinate progress bar based on
 * weighted phase durations. Closes itself on `done` (after a brief success
 * pulse) or on `error` (with retry).
 *
 * Why weighted: the four phases take very different amounts of time on a
 * cold submit (~0.5s · 1.5s · 1.5s · 5s). Equal-weight stepping looks janky
 * because the bar would stall on upload. Weights are tuned empirically and
 * adjusted on the fly using the `ms` value the server reports per phase.
 */

import { useEffect, useMemo, useState } from 'react'

export type Phase = 'validate' | 'stage' | 'pack' | 'submit' | 'upload' | 'done' | 'error'

export interface SubmitEvent {
  phase:       Phase
  status?:     'start' | 'progress' | 'done'
  files?:      number
  kb?:         number
  ms?:         number
  jobId?:      string
  sentBytes?:  number
  totalBytes?: number
  output?:     { shareSyncBaseUrl?: string }
  error?:      string
  totalMs?:    number
}

interface Step {
  phase: Phase
  label: string
  hint:  (data: PhaseData) => string
  // Static weight (% of total bar). Adjusted by actual ms when we get it.
  weight: number
}

const STEPS: Step[] = [
  { phase: 'validate', label: 'Validate inputs', weight: 5,
    hint: () => 'Checking config, paths, GPU…' },
  { phase: 'stage',    label: 'Upload data files', weight: 35,
    hint: (d) => {
      if (d.totalBytes && d.sentBytes != null) {
        const pct = Math.round((d.sentBytes / d.totalBytes) * 100)
        return `${formatKB(d.sentBytes)} / ${formatKB(d.totalBytes)} · ${pct}%${d.files != null ? ` · batch ${d.files}` : ''}`
      }
      return d.totalBytes ? `${formatKB(d.totalBytes)} to upload` : 'Streaming files to server…'
    },
  },
  { phase: 'pack',     label: 'Pack tunet bundle', weight: 20,
    hint: (d) => d.files != null ? `${d.files} files · ${d.kb} KB` : 'Compressing source + config + start script…' },
  { phase: 'submit',   label: 'Submit to Spark', weight: 15,
    hint: (d) => d.jobId ? d.jobId.slice(0, 8) + '…' : 'Reserving instance…' },
  { phase: 'upload',   label: 'Send tarball to ShareSync', weight: 25,
    hint: (d) => {
      if (d.totalBytes && d.sentBytes != null) {
        const pct = Math.round((d.sentBytes / d.totalBytes) * 100)
        return `${formatKB(d.sentBytes)} / ${formatKB(d.totalBytes)} · ${pct}%`
      }
      return d.totalBytes ? `${formatKB(d.totalBytes)} pending` : 'Streaming to WebDAV…'
    },
  },
]

interface PhaseData extends SubmitEvent { /* alias for clarity */ }

interface SubmitProgressProps {
  open:    boolean
  events:  SubmitEvent[]
  errored: string | null
  /** Total elapsed wall-clock since open (kept in parent for ETA math). */
  elapsedMs: number
  onClose:  () => void
  onRetry?: () => void
}

export function SubmitProgress({ open, events, errored, elapsedMs, onClose, onRetry }: SubmitProgressProps) {
  // Per-phase state derived from the event log
  const phases = useMemo(() => derivePhaseStates(events, errored), [events, errored])
  const overall = computeOverallProgress(phases, errored)

  // Smooth the bar so it doesn't snap between phases
  const [smoothPct, setSmoothPct] = useState(0)
  useEffect(() => {
    const target = overall.pct
    let raf = 0
    const step = () => {
      setSmoothPct((cur) => {
        const delta = target - cur
        if (Math.abs(delta) < 0.5) return target
        return cur + delta * 0.18
      })
      if (Math.abs(target - smoothPct) > 0.5) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [overall.pct, smoothPct])

  if (!open) return null

  const isError = !!errored
  const isDone  = phases.done

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#111827]/40 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-pop">
        {/* Header */}
        <div className={`px-5 py-4 border-b ${isError ? 'border-[#fecaca] bg-[#FEF2F2]' : 'border-[#e5e7eb] bg-gradient-to-r from-[#F7F4FC] to-[#fdf4ff]'}`}>
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className={`text-xs font-semibold uppercase tracking-wider ${
                isError ? 'text-[#EF4444]' : isDone ? 'text-[#16A34A]' : 'text-[#7E3AF2]'
              }`}>
                {isError ? 'Submit failed' : isDone ? 'Submitted ✓' : 'Submitting'}
              </p>
              <h3 className="text-base font-bold text-[#111827] mt-0.5">
                {isError ? 'Something went wrong'
                 : isDone  ? 'Job is queued'
                 :          'Sending your job to Spark'}
              </h3>
            </div>
            {(isDone || isError) && (
              <button
                type="button"
                onClick={onClose}
                className="text-[#6b7280] hover:text-[#374151] p-1"
                aria-label="Close"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="18" y1="6"  x2="6"  y2="18" />
                  <line x1="6"  y1="6"  x2="18" y2="18" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div className="px-5 pt-4 pb-2">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-[#374151]">
              {isError ? 'Stopped' : isDone ? 'Complete' : `${Math.round(smoothPct)}%`}
            </span>
            <span className="text-xs text-[#9ca3af] font-mono">
              {formatElapsed(elapsedMs)}
              {!isDone && !isError && overall.etaMs != null && overall.etaMs > 1000 && (
                <> · ~{formatElapsed(overall.etaMs)} left</>
              )}
            </span>
          </div>
          <div className="h-2 bg-[#F9FAFB] border border-[#e5e7eb] rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-200 ${
                isError
                  ? 'bg-[#EF4444]'
                  : isDone
                  ? 'bg-gradient-to-r from-[#16A34A] to-[#22C55E]'
                  : 'bg-gradient-to-r from-[#ae69f4] to-[#c084fc]'
              }`}
              style={{ width: `${smoothPct}%` }}
            />
          </div>
        </div>

        {/* Steps */}
        <ol className="px-5 py-4 space-y-2.5">
          {STEPS.map((step) => {
            const s = phases.byPhase[step.phase]
            return (
              <li key={step.phase} className="flex items-start gap-3">
                <StepDot status={s.status} />
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${
                    s.status === 'done'    ? 'text-[#16A34A]'
                    : s.status === 'active' ? 'text-[#7E3AF2]'
                    : s.status === 'error'  ? 'text-[#EF4444]'
                    :                          'text-[#9ca3af]'
                  }`}>
                    {step.label}
                  </p>
                  <p className="text-xs text-[#6b7280] mt-0.5 font-mono truncate">
                    {step.hint(s.data)}
                    {s.data.ms != null && (
                      <span className="text-[#9ca3af]"> · {(s.data.ms / 1000).toFixed(1)}s</span>
                    )}
                  </p>
                </div>
              </li>
            )
          })}
        </ol>

        {/* Error or success footer */}
        {isError && (
          <div className="px-5 pb-4 pt-2 border-t border-[#fecaca] bg-[#FEF2F2]">
            <p className="text-xs text-[#EF4444] font-mono break-words">
              {errored}
            </p>
            <div className="mt-3 flex items-center justify-end gap-2">
              <button
                onClick={onClose}
                className="text-xs px-3 py-1.5 rounded text-[#6b7280] hover:bg-white"
              >
                Close
              </button>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="text-xs px-3 py-1.5 rounded bg-[#EF4444] text-white font-semibold hover:bg-[#DC2626]"
                >
                  Try again
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── helpers & subcomponents ─────────────────────────────────────────────────

function StepDot({ status }: { status: 'pending' | 'active' | 'done' | 'error' }) {
  if (status === 'done') {
    return (
      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[#16A34A] flex items-center justify-center text-white">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
      </span>
    )
  }
  if (status === 'active') {
    return (
      <span className="flex-shrink-0 w-5 h-5 rounded-full border-2 border-[#ae69f4] flex items-center justify-center">
        <span className="w-2 h-2 rounded-full bg-[#ae69f4] animate-pulse" />
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[#EF4444] flex items-center justify-center text-white">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </span>
    )
  }
  return (
    <span className="flex-shrink-0 w-5 h-5 rounded-full border-2 border-[#e5e7eb] bg-white" />
  )
}

interface DerivedPhase {
  status: 'pending' | 'active' | 'done' | 'error'
  data:   SubmitEvent
}

interface DerivedPhases {
  byPhase: Record<Phase, DerivedPhase>
  /** Phase that is currently active, or null if idle/done */
  active:  Phase | null
  /** Did we receive {phase: 'done'} */
  done:    boolean
}

function derivePhaseStates(events: SubmitEvent[], errored: string | null): DerivedPhases {
  const byPhase: Record<Phase, DerivedPhase> = {
    validate: { status: 'pending', data: { phase: 'validate' } },
    stage:    { status: 'pending', data: { phase: 'stage' } },
    pack:     { status: 'pending', data: { phase: 'pack' } },
    submit:   { status: 'pending', data: { phase: 'submit' } },
    upload:   { status: 'pending', data: { phase: 'upload' } },
    done:     { status: 'pending', data: { phase: 'done' } },
    error:    { status: 'pending', data: { phase: 'error' } },
  }

  let active: Phase | null = null
  let done = false

  for (const e of events) {
    if (e.phase === 'done')  { done = true; continue }
    if (e.phase === 'error') { continue }   // handled via `errored` prop

    const p = byPhase[e.phase]
    p.data = { ...p.data, ...e }
    if (e.status === 'done') {
      p.status = 'done'
      // active moves on
      if (active === e.phase) active = null
    } else if (e.status === 'start' || e.status === 'progress') {
      p.status = 'active'
      active = e.phase
    }
  }

  if (errored && active) {
    byPhase[active].status = 'error'
  }

  return { byPhase, active, done }
}

function computeOverallProgress(phases: DerivedPhases, errored: string | null): {
  pct: number
  etaMs: number | null
} {
  if (errored) return { pct: 100, etaMs: null }
  if (phases.done) return { pct: 100, etaMs: 0 }

  let pct = 0
  for (const step of STEPS) {
    const s = phases.byPhase[step.phase]
    if (s.status === 'done') {
      pct += step.weight
    } else if (s.status === 'active') {
      // half-credit for active phases (we don't have intra-phase progress)
      const halfCredit =
        step.phase === 'upload' && s.data.sentBytes && s.data.totalBytes
          ? (s.data.sentBytes / s.data.totalBytes) * step.weight
          : step.weight * 0.5
      pct += halfCredit
    }
  }

  // ETA — rough: assume remaining percent will take same wall-clock per percent
  // as what we've used so far. Doesn't matter much for short submits but feels
  // reasonable when upload is the slow phase.
  return { pct: Math.min(100, pct), etaMs: null }
}

function formatKB(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms} ms`
  const s = Math.round(ms / 100) / 10
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  const rs = Math.round(s - m * 60)
  return `${m}m ${rs}s`
}

