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

import { useEffect, useRef, useState } from 'react'
import type { SparkJob } from '@/lib/spark-types'
import { jobLabel } from '@/lib/spark-types'
import type { SourceJobRef } from '@/lib/spark-form-state'
import { uploadCheckpoint } from '@/lib/upload-stage'

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
  // Two source paths: a prior Spark job (existing behavior) or a .pth file
  // the user has on their machine (new — for users who trained off-Spark
  // and want to continue training on Spark). The 'local' path skips all the
  // Spark API plumbing and just uploads the .pth via /api/spark/upload-stage
  // with role='checkpoint'.
  const [sourceMode, setSourceMode] = useState<'spark' | 'local'>(
    value?.localCheckpointStageId ? 'local' : 'spark'
  )

  const [jobs,    setJobs]    = useState<SparkJob[] | null>(null)
  const [jobsErr, setJobsErr] = useState<string | null>(null)
  const [pickedJobId, setPickedJobId] = useState<string>(value?.jobId ?? '')

  const [ckpts,    setCkpts]    = useState<ChecksResponse | null>(null)
  const [ckptsErr, setCkptsErr] = useState<string | null>(null)
  const [loading,  setLoading]  = useState(false)

  // Local-upload state — populated as the user picks + uploads a .pth.
  const [localFile,       setLocalFile]       = useState<File | null>(null)
  const [localUploading,  setLocalUploading]  = useState(false)
  const [localError,      setLocalError]      = useState<string | null>(null)
  // Persist the uploaded handle so the user can swap files / fix the name
  // without losing the staged upload.
  const [localUploaded,   setLocalUploaded]   = useState<{ stageId: string; filename: string; bytes: number } | null>(
    value?.localCheckpointStageId
      ? { stageId: value.localCheckpointStageId, filename: value.checkpointName, bytes: 0 }
      : null
  )
  const localInputRef = useRef<HTMLInputElement | null>(null)

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

  // Note: we used to auto-clear pickedJobId whenever the parent's value went
  // null, but that fired in the wrong window — when a user picks a job, our
  // local state flips first and the parent's value catches up async after the
  // checkpoints fetch resolves. The auto-clear would reset the dropdown mid-
  // pick. Mode flip back to 'new' unmounts us entirely (gate at new/page.tsx),
  // so we don't actually need to react to value going null while mounted.

  // Filter to jobs that *probably* have output: terminal-success or running.
  // We don't strictly enforce — Spark's status semantics are fuzzy, and the
  // checkpoints endpoint will report `pending: true` for jobs with no .pth yet.
  const candidateJobs = (jobs ?? []).filter(j => {
    // Hide jobs that never reached a state that could have produced output.
    return !['queued', 'provisioning'].includes(j.status) || j.last_agent_heartbeat_at
  })

  async function handleLocalFileSelected(file: File) {
    setLocalFile(file)
    setLocalError(null)
    if (!file.name.endsWith('.pth')) {
      setLocalError('Pick a .pth file (PyTorch checkpoint).')
      return
    }
    setLocalUploading(true)
    try {
      const result = await uploadCheckpoint(file)
      setLocalUploaded(result)
      // Tell the parent — `jobId: 'local-upload'` is the sentinel the
      // training-jobs route uses to know not to call the Spark API.
      // preset/gpuKey are null because we can't infer them from a local file.
      onChange(
        {
          jobId:          'local-upload',
          jobLabel:       `Local: ${result.filename}`,
          checkpointName: result.filename,
          localCheckpointStageId: result.stageId,
        },
        { preset: null, gpuKey: null },
      )
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : 'Upload failed')
      setLocalUploaded(null)
      onChange(null, { preset: null, gpuKey: null })
    } finally {
      setLocalUploading(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Source-mode tabs */}
      <div className="inline-flex rounded-lg border border-[#e5e7eb] bg-white p-0.5 text-xs font-semibold">
        <button
          type="button"
          onClick={() => {
            setSourceMode('spark')
            // Switching away from local clears the local picker but keeps
            // the Spark selection (if any) intact — let onChange handle the
            // current-value update when jobs load.
            if (sourceMode !== 'spark' && value?.localCheckpointStageId) {
              onChange(null, { preset: null, gpuKey: null })
            }
          }}
          className={`px-3 py-1.5 rounded-md transition-colors ${
            sourceMode === 'spark' ? 'bg-[#7E3AF2] text-white' : 'text-[#6b7280] hover:bg-[#F9FAFB]'
          }`}
        >
          From Spark job
        </button>
        <button
          type="button"
          onClick={() => {
            setSourceMode('local')
            if (sourceMode !== 'local' && value && !value.localCheckpointStageId) {
              // Clear the Spark selection so the form doesn't submit both
              setPickedJobId('')
              setCkpts(null)
              onChange(null, { preset: null, gpuKey: null })
            }
          }}
          className={`px-3 py-1.5 rounded-md transition-colors ${
            sourceMode === 'local' ? 'bg-[#7E3AF2] text-white' : 'text-[#6b7280] hover:bg-[#F9FAFB]'
          }`}
        >
          Upload local .pth
        </button>
      </div>

      {/* Local .pth upload */}
      {sourceMode === 'local' && (
        <div className="pl-3 border-l-2 border-[#e9d5ff] space-y-2">
          <p className="text-[11px] text-[#6b7280] leading-relaxed">
            For continuing a model you trained on your own machine. The .pth
            file (typically 100-500 MB) uploads to a staging area on the
            tunet-web server, then ships in the Spark job tarball. The new run
            seeds your checkpoint at the start so train.py picks it up via the
            same auto-resume path as a Spark-source resume.
          </p>
          <div className="flex items-center gap-2">
            <input
              ref={localInputRef}
              type="file"
              accept=".pth"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleLocalFileSelected(f)
              }}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => localInputRef.current?.click()}
              disabled={localUploading}
              className="px-3 py-1.5 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff] disabled:opacity-50"
            >
              {localUploading ? 'Uploading…' : (localUploaded ? 'Replace .pth' : 'Browse for .pth')}
            </button>
            {localUploaded && !localUploading && (
              <span className="text-xs text-[#16A34A] font-mono">
                ✓ {localUploaded.filename}
                {localUploaded.bytes > 0 && ` (${formatBytes(localUploaded.bytes)})`}
              </span>
            )}
            {localFile && localUploading && (
              <span className="text-xs text-[#6b7280] font-mono">
                Uploading {localFile.name} ({formatBytes(localFile.size)})…
              </span>
            )}
          </div>
          {localError && (
            <p className="text-xs text-[#EF4444]">{localError}</p>
          )}
          {localUploaded && !localUploading && mode === 'resume' && (
            <p className="text-[11px] text-[#6b7280] leading-relaxed">
              The new run will write into <code className="bg-[#F9FAFB] px-1 rounded">/output/&lt;job-name&gt;</code>{' '}
              with this checkpoint seeded so train.py can pick up where local training left off.
              <strong className="text-[#D97706]"> Preset / model size / loss must match what you trained
              locally</strong> — train.py refuses to load a checkpoint with mismatched architecture.
            </p>
          )}
          {localUploaded && !localUploading && mode === 'finetune' && (
            <p className="text-[11px] text-[#6b7280] leading-relaxed">
              Weights are loaded from your local .pth; optimizer + step counter reset on the
              Spark side. Preset / model can differ from what you trained locally as long as
              the architecture shape matches.
            </p>
          )}
        </div>
      )}

      {/* Job picker (Spark-source path) */}
      {sourceMode === 'spark' && (<>
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
      </>)}
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
