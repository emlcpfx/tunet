'use client'

/**
 * Shared multi-select jobs table used by /demo/dashboard ("Recent Jobs") and
 * /demo/jobs (full list).
 *
 * Selection rules:
 *   - All rows are selectable (terminal jobs included; we just can't cancel
 *     them, but we CAN hide them from the list).
 *   - Bulk action picks the right verb based on the selection mix:
 *       all active   → "Cancel N jobs"
 *       all terminal → "Hide N jobs"
 *       mixed        → "Cancel + Hide N jobs" (cancels the active ones first
 *                       and hides everything once they're terminal-bound)
 *
 * Hiding writes to localStorage via lib/hidden-jobs.ts. Spark Compute v1 has
 * no delete endpoint, so this is the closest we can get to "remove from the
 * dashboard". A "Show hidden (N)" toggle below the table reveals them again.
 */

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import {
  type SparkJob, ACTIVE_STATUSES, jobLabel, jobMode, jobRuntimeMs, formatRuntime, derivedStatus,
} from '@/lib/spark-types'
import { hideJobs, restoreJobs, useHiddenJobs } from '@/lib/hidden-jobs'
import { SparkStatusBadge } from './status-badge'

interface JobsTableProps {
  jobs:           SparkJob[]
  /** "Started" column formatter — dashboard wants relative ("3m ago"), list wants absolute. */
  startedFormat:  (j: SparkJob) => string
  /** Called after a successful bulk action so the parent can refresh its list. */
  onAfterAction?: () => void
}

interface ActionOutcome {
  cancelled: string[]
  hidden:    string[]
  failed:    { id: string; error: string }[]
}

export function JobsTable({ jobs, startedFormat, onAfterAction }: JobsTableProps) {
  const hidden = useHiddenJobs()
  const [showHidden, setShowHidden] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<ActionOutcome | null>(null)

  // Filter hidden jobs out unless the user asked to see them.
  const visibleJobs = useMemo(() => {
    if (showHidden) return jobs
    return jobs.filter(j => !hidden.has(j.id))
  }, [jobs, hidden, showHidden])

  // Selections become invalid if a job disappears (e.g. polled out of the
  // list). Drop them so the action bar count stays honest.
  const validSelected = useMemo(() => {
    const present = new Set(visibleJobs.map(j => j.id))
    const next = new Set<string>()
    for (const id of selected) if (present.has(id)) next.add(id)
    return next
  }, [selected, visibleJobs])

  // Classify selection — drives the bulk-action verb.
  const selectedJobs = visibleJobs.filter(j => validSelected.has(j.id))
  const activeCount   = selectedJobs.filter(j => ACTIVE_STATUSES.has(j.status)).length
  const terminalCount = selectedJobs.length - activeCount

  const allVisibleSelected = visibleJobs.length > 0 && visibleJobs.every(j => validSelected.has(j.id))
  const someSelected       = validSelected.size > 0 && !allVisibleSelected

  const hiddenCount = useMemo(() => {
    // Only count hidden jobs that actually exist in the current list (rather
    // than stale IDs from old jobs no longer returned by the API).
    let n = 0
    for (const j of jobs) if (hidden.has(j.id)) n++
    return n
  }, [jobs, hidden])

  function toggle(id: string) {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }

  function toggleAll() {
    if (allVisibleSelected) setSelected(new Set())
    else setSelected(new Set(visibleJobs.map(j => j.id)))
  }

  // ── Bulk action ──────────────────────────────────────────────────────────
  // Smart verb:
  //   pure terminal → Hide (just localStorage write)
  //   pure active   → Cancel (POST cancel)
  //   mixed         → Cancel actives, then hide ALL — best of both
  async function runBulkAction() {
    const active   = selectedJobs.filter(j =>  ACTIVE_STATUSES.has(j.status)).map(j => j.id)
    const terminal = selectedJobs.filter(j => !ACTIVE_STATUSES.has(j.status)).map(j => j.id)
    if (active.length + terminal.length === 0) return

    // Confirm — copy depends on what we're about to do
    const parts: string[] = []
    if (active.length)   parts.push(`Cancel ${active.length} active job${active.length === 1 ? '' : 's'} (SIGTERM)`)
    if (terminal.length) parts.push(`Hide ${terminal.length} terminal job${terminal.length === 1 ? '' : 's'} from this dashboard`)
    const confirmMsg = parts.join('\nAND\n') + '\n\nProceed?'
    if (!window.confirm(confirmMsg)) return

    setBusy(true)
    setOutcome(null)

    // Cancel actives in parallel
    const cancelResults = await Promise.all(active.map(async (id) => {
      try {
        const res = await fetch(`/api/spark/jobs/${id}/cancel`, { method: 'POST' })
        if (!res.ok) {
          const j = await res.json().catch(() => ({}))
          return { id, ok: false as const, error: j.error ?? `HTTP ${res.status}` }
        }
        return { id, ok: true as const }
      } catch (e) {
        return { id, ok: false as const, error: e instanceof Error ? e.message : 'Network error' }
      }
    }))

    // Hide terminals (always succeeds — pure localStorage)
    if (terminal.length > 0) hideJobs(terminal)

    setBusy(false)

    const cancelled = cancelResults.filter(r => r.ok).map(r => r.id)
    const failed    = cancelResults.filter(r => !r.ok).map(r => ({
      id:    r.id,
      error: 'error' in r ? r.error : 'unknown',
    }))
    setOutcome({ cancelled, hidden: terminal, failed })
    setSelected(new Set())
    onAfterAction?.()
  }

  // Restore a single hidden job
  function restore(id: string) {
    restoreJobs([id])
  }

  // Auto-clear the outcome banner after 5s
  useEffect(() => {
    if (!outcome) return
    const id = window.setTimeout(() => setOutcome(null), 6000)
    return () => window.clearTimeout(id)
  }, [outcome])

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-3">
      {/* Action bar */}
      {validSelected.size > 0 && (
        <div className="flex items-center justify-between gap-3 px-4 py-2.5 bg-[#F7F4FC] border border-[#e9d5ff] rounded-lg">
          <p className="text-sm text-[#374151]">
            <span className="font-semibold">{validSelected.size}</span> selected
            {activeCount > 0 && terminalCount > 0 && (
              <span className="text-xs text-[#6b7280] ml-2">
                ({activeCount} active · {terminalCount} terminal)
              </span>
            )}
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSelected(new Set())}
              disabled={busy}
              className="px-3 py-1.5 text-xs font-medium text-[#6b7280] hover:text-[#374151] disabled:opacity-50"
            >
              Clear
            </button>
            <BulkActionButton
              busy={busy}
              activeCount={activeCount}
              terminalCount={terminalCount}
              onClick={runBulkAction}
            />
          </div>
        </div>
      )}

      {/* Outcome banner */}
      {outcome && (outcome.cancelled.length + outcome.hidden.length + outcome.failed.length > 0) && (
        <div className={`px-4 py-2.5 rounded-lg text-sm border ${
          outcome.failed.length === 0
            ? 'bg-[#F0FDF4] border-[#bbf7d0] text-[#166534]'
            : 'bg-[#FEF2F2] border-[#fecaca] text-[#7F1D1D]'
        }`}>
          {outcome.cancelled.length > 0 && (
            <p>Cancelled {outcome.cancelled.length} job{outcome.cancelled.length === 1 ? '' : 's'}.</p>
          )}
          {outcome.hidden.length > 0 && (
            <p>
              Hid {outcome.hidden.length} job{outcome.hidden.length === 1 ? '' : 's'} from this dashboard.{' '}
              <button
                type="button"
                onClick={() => { restoreJobs(outcome.hidden); setOutcome(null) }}
                className="underline font-medium hover:text-[#7E3AF2]"
              >
                Undo
              </button>
            </p>
          )}
          {outcome.failed.length > 0 && (
            <ul className="mt-1 text-xs font-mono space-y-0.5">
              {outcome.failed.map(f => (
                <li key={f.id}>{f.id.slice(0, 8)}… — {f.error}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Table */}
      <div className="bg-white border border-[#e5e7eb] rounded-xl overflow-hidden card-shadow">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#e5e7eb] bg-[#F9FAFB]">
              <th className="w-10 px-4 py-3">
                <input
                  type="checkbox"
                  aria-label="Select all jobs"
                  checked={allVisibleSelected}
                  ref={el => { if (el) el.indeterminate = someSelected }}
                  onChange={toggleAll}
                  disabled={visibleJobs.length === 0}
                  className="h-3.5 w-3.5 rounded border-[#D1D5DB] accent-[#7E3AF2] disabled:opacity-40 cursor-pointer"
                />
              </th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Job</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">GPU</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Status</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Started</th>
              <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Duration</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {visibleJobs.map((j, i) => {
              const isSelected = validSelected.has(j.id)
              const isHidden   = hidden.has(j.id)
              return (
                <tr
                  key={j.id}
                  className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} ${
                    isSelected ? 'bg-[#F7F4FC]' : 'hover:bg-[#F9FAFB]'
                  } ${isHidden ? 'opacity-50' : ''} transition-colors`}
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      aria-label={`Select ${jobLabel(j)}`}
                      checked={isSelected}
                      onChange={() => toggle(j.id)}
                      className="h-3.5 w-3.5 rounded border-[#D1D5DB] accent-[#7E3AF2] cursor-pointer"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <Link href={`/demo/jobs/${j.id}`} className="font-mono text-xs text-[#111827] hover:text-[#ae69f4]">
                      {jobLabel(j)}
                    </Link>
                    {(() => {
                      const m = jobMode(j)
                      if (m.mode === 'new') return null
                      // Tag jobs that started from a prior checkpoint so the
                      // list immediately tells you "this is a continuation".
                      return (
                        <span
                          className="ml-2 text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-[#e9d5ff] text-[#7E3AF2] bg-[#F7F4FC]"
                          title={`${m.mode === 'resume' ? 'Resume' : 'Fine-tune'} of job ${m.sourceJobId ?? '—'}`}
                        >
                          {m.mode === 'resume' ? 'resume' : 'fine-tune'}
                        </span>
                      )
                    })()}
                    {isHidden && (
                      <span className="ml-2 text-[10px] uppercase tracking-wider text-[#9ca3af]">hidden</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{j.instance_type_name ?? '—'}</td>
                  <td className="px-4 py-3"><SparkStatusBadge status={j.status} liveOverride={derivedStatus(j)} /></td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{startedFormat(j)}</td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{formatRuntime(jobRuntimeMs(j))}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-3">
                      {isHidden ? (
                        <button
                          type="button"
                          onClick={() => restore(j.id)}
                          className="text-xs text-[#7E3AF2] hover:underline"
                        >
                          Restore
                        </button>
                      ) : (
                        <Link
                          href={`/demo/jobs/new?clone=${j.id}`}
                          className="text-xs text-[#6b7280] hover:text-[#7E3AF2]"
                          title="Start a new job pre-filled from this one"
                        >
                          Clone
                        </Link>
                      )}
                      <Link href={`/demo/jobs/${j.id}`} className="text-xs text-[#ae69f4] hover:underline">View →</Link>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Hidden-jobs affordance */}
      {hiddenCount > 0 && (
        <div className="flex items-center justify-end gap-2 text-xs text-[#6b7280]">
          <span>{hiddenCount} hidden</span>
          <button
            type="button"
            onClick={() => setShowHidden(!showHidden)}
            className="text-[#7E3AF2] hover:underline font-medium"
          >
            {showHidden ? 'Hide them again' : 'Show hidden'}
          </button>
        </div>
      )}
    </div>
  )
}

// ── Subcomponents ──────────────────────────────────────────────────────────

function BulkActionButton({ busy, activeCount, terminalCount, onClick }: {
  busy:          boolean
  activeCount:   number
  terminalCount: number
  onClick:       () => void
}) {
  // Pick verb + style
  let label: string
  let style: 'danger' | 'neutral'

  if (activeCount > 0 && terminalCount > 0) {
    label = busy ? 'Working…' : `Cancel ${activeCount} · Hide ${terminalCount}`
    style = 'danger'
  } else if (activeCount > 0) {
    label = busy ? 'Cancelling…' : `Cancel ${activeCount} job${activeCount === 1 ? '' : 's'}`
    style = 'danger'
  } else {
    label = busy ? 'Hiding…' : `Hide ${terminalCount} job${terminalCount === 1 ? '' : 's'}`
    style = 'neutral'
  }

  const cls = style === 'danger'
    ? 'bg-[#FEF2F2] text-[#EF4444] border-[#fecaca] hover:bg-[#FEE2E2]'
    : 'bg-white text-[#374151] border-[#e5e7eb] hover:bg-[#F9FAFB] hover:border-[#7E3AF2] hover:text-[#7E3AF2]'

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className={`px-3 py-1.5 rounded-md text-xs font-semibold border disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${cls}`}
    >
      {label}
    </button>
  )
}
