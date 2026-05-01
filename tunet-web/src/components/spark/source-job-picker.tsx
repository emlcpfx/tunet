'use client'

/**
 * Source job + checkpoint picker for Resume / Fine-tune modes.
 *
 * Two-step UI:
 *   1. Pick a prior job   — dropdown of recent jobs (id + label + status).
 *   2. Pick a checkpoint  — auto = latest (resume); explicit picker (fine-tune).
 *
 * Surfaces `pending: true` from the checkpoints API as a friendly "no
 * checkpoints uploaded yet" hint instead of an error.
 */

import { useEffect, useState } from 'react'
import type { SparkJob } from '@/lib/spark-types'
import { jobLabel } from '@/lib/spark-types'
import type { SourceJobRef } from '@/lib/spark-form-state'

interface CheckpointEntry {
  name:     string
  size:     number
  modified: string | null
}

interface ChecksResponse {
  job:    { id: string; label: string; preset: string | null; gpuKey: string | null; outputSubdir: string | null }
  latest: CheckpointEntry | null
  epochs: CheckpointEntry[]
  pending?: boolean
}

interface SourceJobPickerProps {
  /** 'resume' shows the latest auto-pick + warning about preset lock; 'finetune' shows the full epoch picker. */
  mode:     'resume' | 'finetune'
  value:    SourceJobRef | null
  onChange: (ref: SourceJobRef | null, meta: { preset: string | null; gpuKey: string | null }) => void
}

export function SourceJobPicker({ mode, value, onChange }: SourceJobPickerProps) {
  const [jobs,    setJobs]    = useState<SparkJob[] | null>(null)
  const [jobsErr, setJobsErr] = useState<string | null>(null)
  const [pickedJobId, setPickedJobId] = useState<string>(value?.jobId ?? '')

  const [ckpts,    setCkpts]    = useState<ChecksResponse | null>(null)
  const [ckptsErr, setCkptsErr] = useState<string | null>(null)
  const [loading,  setLoading]  = useState(false)

  // Load list of jobs once on mount. We keep all of them and filter client-side
  // — the API response is small (a few KB even at hundreds of jobs).
  useEffect(() => {
    let cancelled = false
    fetch('/api/spark/jobs', { cache: 'no-store' })
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(d => { if (!cancelled) setJobs((d.jobs as SparkJob[]) ?? []) })
      .catch(e => { if (!cancelled) setJobsErr(e instanceof Error ? e.message : 'Load failed') })
    return () => { cancelled = true }
  }, [])

  // When a job is picked, fetch its checkpoints.
  useEffect(() => {
    if (!pickedJobId) { setCkpts(null); setCkptsErr(null); return }
    let cancelled = false
    setLoading(true)
    setCkpts(null); setCkptsErr(null)
    fetch(`/api/spark/jobs/${pickedJobId}/checkpoints`, { cache: 'no-store' })
      .then(r => r.ok ? r.json() : r.json().then(j => Promise.reject(new Error(j.error ?? `HTTP ${r.status}`))))
      .then((d: ChecksResponse) => {
        if (cancelled) return
        setCkpts(d)
        // Auto-fill: resume = latest; finetune = latest as a sensible default
        // (user can change in the dropdown).
        const auto = d.latest?.name
        if (auto) {
          onChange(
            { jobId: d.job.id, jobLabel: d.job.label, checkpointName: auto },
            { preset: d.job.preset, gpuKey: d.job.gpuKey },
          )
        } else {
          // No checkpoints yet — clear so the form blocks submit until ready.
          onChange(null, { preset: d.job.preset, gpuKey: d.job.gpuKey })
        }
      })
      .catch(e => { if (!cancelled) setCkptsErr(e instanceof Error ? e.message : 'Load failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onChange not stable; we only want to re-run on jobId change
  }, [pickedJobId])

  // If parent clears `value` (e.g. mode flip back to 'new'), keep our local
  // dropdown in sync so re-entering the mode shows a clean state.
  useEffect(() => {
    if (!value && pickedJobId) setPickedJobId('')
  }, [value, pickedJobId])

  // Filter to jobs that *probably* have output: terminal-success or running.
  // We don't strictly enforce — Spark's status semantics are fuzzy, and the
  // checkpoints endpoint will report `pending: true` for jobs with no .pth yet.
  const candidateJobs = (jobs ?? []).filter(j => {
    // Hide jobs that never reached a state that could have produced output.
    return !['queued', 'provisioning'].includes(j.status) || j.last_agent_heartbeat_at
  })

  return (
    <div className="space-y-3">
      {/* Job picker */}
      <div>
        <label className="block text-xs font-semibold text-[#374151] mb-1.5">
          {mode === 'resume' ? 'Resume which job?' : 'Fine-tune from which job?'}
        </label>
        {jobsErr ? (
          <p className="text-xs text-[#EF4444]">Could not load jobs: {jobsErr}</p>
        ) : !jobs ? (
          <p className="text-xs text-[#9ca3af]">Loading jobs…</p>
        ) : candidateJobs.length === 0 ? (
          <p className="text-xs text-[#9ca3af]">No prior jobs found yet.</p>
        ) : (
          <select
            value={pickedJobId}
            onChange={(e) => setPickedJobId(e.target.value)}
            className="w-full max-w-md text-sm font-mono px-3 py-2 border border-[#e5e7eb] rounded-lg bg-white focus:border-[#7E3AF2] focus:outline-none"
          >
            <option value="">— Pick a job —</option>
            {candidateJobs.map(j => (
              <option key={j.id} value={j.id}>
                {jobLabel(j)} · {j.status}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Checkpoint picker / status */}
      {pickedJobId && (
        <div className="pl-3 border-l-2 border-[#e9d5ff] space-y-2">
          {loading && <p className="text-xs text-[#9ca3af]">Loading checkpoints…</p>}
          {ckptsErr && <p className="text-xs text-[#EF4444]">Could not load checkpoints: {ckptsErr}</p>}
          {ckpts && ckpts.pending && (
            <p className="text-xs text-[#D97706]">
              No checkpoints uploaded for this job yet. Wait for it to write its first
              checkpoint, or pick a different job.
            </p>
          )}
          {ckpts && !ckpts.pending && !ckpts.latest && ckpts.epochs.length === 0 && (
            <p className="text-xs text-[#D97706]">
              This job has no .pth files in its output dir.
            </p>
          )}
          {ckpts && (ckpts.latest || ckpts.epochs.length > 0) && (
            <CheckpointPicker
              mode={mode}
              ckpts={ckpts}
              value={value?.checkpointName ?? null}
              onChange={(name) => onChange(
                { jobId: ckpts.job.id, jobLabel: ckpts.job.label, checkpointName: name },
                { preset: ckpts.job.preset, gpuKey: ckpts.job.gpuKey },
              )}
            />
          )}

          {/* Mode-specific hints */}
          {ckpts && value && mode === 'resume' && (
            <p className="text-[11px] text-[#6b7280] leading-relaxed">
              The new run will write into <code className="bg-[#F9FAFB] px-1 rounded">/output/{ckpts.job.outputSubdir}</code>{' '}
              (same as the source job) so the trainer can pick up where it left off.
              Preset, model size, loss, and mask-input setting must match — train.py
              will refuse to load if they differ.
            </p>
          )}
          {ckpts && value && mode === 'finetune' && (
            <p className="text-[11px] text-[#6b7280] leading-relaxed">
              Weights are loaded; optimizer + step counter reset. New output dir, fresh
              training log. Preset / model can differ from the source as long as the
              architecture shape matches.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

interface CheckpointPickerProps {
  mode:     'resume' | 'finetune'
  ckpts:    ChecksResponse
  value:    string | null
  onChange: (name: string) => void
}

function CheckpointPicker({ mode, ckpts, value, onChange }: CheckpointPickerProps) {
  const all = [
    ...(ckpts.latest ? [{ ...ckpts.latest, isLatest: true }]  : []),
    ...ckpts.epochs.map(e => ({ ...e, isLatest: false })),
  ]
  // Resume always uses latest — show as a read-only line.
  if (mode === 'resume') {
    if (!ckpts.latest) {
      return <p className="text-xs text-[#D97706]">No latest .pth found in this job&apos;s output.</p>
    }
    return (
      <div className="text-xs">
        <span className="text-[#6b7280]">Latest checkpoint: </span>
        <code className="bg-[#F9FAFB] px-1.5 py-0.5 rounded font-mono text-[#374151]">{ckpts.latest.name}</code>
        <span className="text-[#9ca3af]"> · {formatBytes(ckpts.latest.size)} · {formatTime(ckpts.latest.modified)}</span>
      </div>
    )
  }
  // Fine-tune: full picker.
  return (
    <div className="space-y-1.5">
      <label className="block text-xs font-semibold text-[#374151]">Checkpoint</label>
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        className="w-full max-w-md text-xs font-mono px-3 py-2 border border-[#e5e7eb] rounded-lg bg-white focus:border-[#7E3AF2] focus:outline-none"
      >
        <option value="" disabled>— Pick a .pth —</option>
        {all.map(c => (
          <option key={c.name} value={c.name}>
            {c.isLatest ? '★ ' : ''}{c.name}  ·  {formatBytes(c.size)}  ·  {formatTime(c.modified)}
          </option>
        ))}
      </select>
      <p className="text-[10px] text-[#9ca3af]">★ = latest. Per-epoch checkpoints let you fine-tune from a specific point in training.</p>
    </div>
  )
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}
