'use client'

/**
 * Pre-log "what's happening?" timeline for a Spark Compute job.
 *
 * Shown above the log stream panel. Polls /api/spark/jobs/:id every 2.5s
 * and surfaces the lifecycle phases the agent is moving through:
 *
 *   queued                  · job sitting in the scheduler queue
 *   provisioning            · warm-pool instance reserved, agent booting
 *   pulling-image           · `docker pull` (large pytorch image is ~5min cold)
 *   container-starting      · image extracted, container starting
 *   running (waiting logs)  · container up, first log line not in yet
 *   running (logs)          · we have logs — caller hides this widget
 *
 * Most of these phases aren't directly named in the job-detail response;
 * we infer them from timestamps + heartbeat:
 *
 *   created_at                 → queued
 *   started_provisioning_at    → provisioning / pulling image
 *   last_agent_heartbeat_at    → container alive (heartbeat ticks during pull)
 *   started_running_at         → container started
 *   cancel_requested_at        → cancel pending
 *   terminal_at + status       → done / failed / cancelled
 */

import { useEffect, useState } from 'react'
import type { SparkJob } from '@/lib/spark-types'

interface ProvisioningTimelineProps {
  jobId:        string
  /** Initial server-rendered job (avoid first-poll flash) */
  initialJob:   SparkJob
  /** Called once we observe logs (caller can collapse the timeline) */
  onLogsStart?: () => void
  /** True if any log lines have arrived in the parent component */
  hasLogs:      boolean
}

type PhaseStatus = 'pending' | 'active' | 'done' | 'error'

interface Phase {
  key:   string
  label: string
  hint:  string
}

const PHASES: Phase[] = [
  { key: 'queued',       label: 'Queued',           hint: 'Waiting for an instance to schedule.' },
  { key: 'provisioning', label: 'Provisioning EC2', hint: 'Reserving a warm instance and booting the agent.' },
  { key: 'pulling',      label: 'Pulling image',    hint: 'Downloading the PyTorch container (large; can take 1–3 min cold).' },
  { key: 'starting',     label: 'Starting container', hint: 'Extracting input tarball and launching your start script.' },
  { key: 'running',      label: 'Running',          hint: 'Container is up. Waiting for first log line…' },
]

export function ProvisioningTimeline({ jobId, initialJob, onLogsStart, hasLogs }: ProvisioningTimelineProps) {
  const [job, setJob] = useState<SparkJob>(initialJob)
  const [pollError, setPollError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)   // forces relative-time labels to update

  // Poll job state — slow down once we have evidence the container is up.
  // Once logs are flowing we mostly only need to detect the terminal status.
  useEffect(() => {
    const interval = (job.status === 'running' || hasLogs) ? 5000 : 2500
    const id = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/spark/jobs/${jobId}`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (data.job) setJob(data.job as SparkJob)
        setPollError(null)
      } catch (e) {
        setPollError(e instanceof Error ? e.message : 'poll failed')
      }
    }, interval)
    return () => window.clearInterval(id)
  }, [jobId, job.status, hasLogs])

  // Tick clock so "12s ago" updates without a poll
  useEffect(() => {
    const id = window.setInterval(() => setTick(t => t + 1), 1000)
    return () => window.clearInterval(id)
  }, [])

  // Once logs arrive, notify parent. We don't auto-hide here so the parent
  // can choose to keep us visible or collapse.
  useEffect(() => {
    if (hasLogs && onLogsStart) onLogsStart()
  }, [hasLogs, onLogsStart])

  const phaseStates = derivePhases(job, hasLogs)
  const isTerminal = ['succeeded', 'completed', 'failed', 'cancelled'].includes(job.status)

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-[#374151] uppercase tracking-wider">
          Provisioning {(job.status === 'running' || hasLogs) ? '· Container ready' : ''}
        </p>
        <span className="text-xs text-[#9ca3af] font-mono" data-tick={tick}>
          {pollError ? <span className="text-[#D97706]">{pollError}</span> : 'live'}
        </span>
      </div>

      <ol className="space-y-2.5">
        {PHASES.map(p => {
          const ps = phaseStates[p.key] ?? { status: 'pending' as PhaseStatus }
          return (
            <li key={p.key} className="flex items-start gap-3">
              <PhaseDot status={ps.status} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <p className={`text-sm font-medium ${
                    ps.status === 'done'   ? 'text-[#16A34A]' :
                    ps.status === 'active' ? 'text-[#7E3AF2]' :
                    ps.status === 'error'  ? 'text-[#EF4444]' :
                                             'text-[#9ca3af]'
                  }`}>
                    {p.label}
                  </p>
                  {ps.completedAt && (
                    <span className="text-[10px] font-mono text-[#9ca3af]">
                      {formatRelative(ps.completedAt)}
                    </span>
                  )}
                  {ps.elapsedMs != null && ps.status === 'active' && (
                    <span className="text-[10px] font-mono text-[#7E3AF2]">
                      {formatDur(ps.elapsedMs)}
                    </span>
                  )}
                </div>
                {ps.status === 'active' && (
                  <p className="text-xs text-[#6b7280] mt-0.5">{p.hint}</p>
                )}
              </div>
            </li>
          )
        })}
      </ol>

      {/* Show resource info once we have it */}
      {job.gpu_name && (
        <div className="mt-3 pt-3 border-t border-[#F3F4F6] flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[#6b7280]">
          <span><span className="text-[#9ca3af]">GPU:</span> {job.gpu_name}</span>
          {job.cuda_version && <span><span className="text-[#9ca3af]">CUDA:</span> {job.cuda_version}</span>}
          {job.driver_version && <span><span className="text-[#9ca3af]">Driver:</span> {job.driver_version}</span>}
          {job.instance_type_name && <span className="font-mono"><span className="text-[#9ca3af]">SKU:</span> {job.instance_type_name}</span>}
        </div>
      )}

      {isTerminal && job.error_message && (
        <div className="mt-3 px-3 py-2 bg-[#FEF2F2] border border-[#fecaca] rounded text-xs text-[#7F1D1D] font-mono">
          {job.error_message}
        </div>
      )}
    </div>
  )
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function PhaseDot({ status }: { status: PhaseStatus }) {
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
  return <span className="flex-shrink-0 w-5 h-5 rounded-full border-2 border-[#e5e7eb] bg-white" />
}

interface PhaseState {
  status:       PhaseStatus
  completedAt?: string   // ISO string when this phase finished
  elapsedMs?:   number   // ms in this phase (only set for active)
}

/**
 * Map the job's snake_case timestamps to per-phase state.
 */
function derivePhases(job: SparkJob, hasLogs: boolean): Record<string, PhaseState> {
  const now = Date.now()
  const created   = job.created_at              ? Date.parse(job.created_at)              : null
  const provStart = job.started_provisioning_at ? Date.parse(job.started_provisioning_at) : null
  const heartbeat = job.last_agent_heartbeat_at ? Date.parse(job.last_agent_heartbeat_at) : null
  const runStart  = job.started_running_at      ? Date.parse(job.started_running_at)      : null
  const terminal  = job.terminal_at             ? Date.parse(job.terminal_at)             : null
  const cancelReq = job.cancel_requested_at     ? Date.parse(job.cancel_requested_at)     : null

  const isFailed = job.status === 'failed' || (job.status === 'cancelled' && !cancelReq)
  const isTerminalSuccess = job.status === 'succeeded' || job.status === 'completed'
  const isCancelled = job.status === 'cancelled' && !!cancelReq

  // ── Failure-phase classification ──────────────────────────────────────────
  // Spark populates `gpu_name` / `cuda_version` from SKU metadata, NOT from a
  // heartbeat — so they're not a reliable signal that the container started.
  // Use error_code instead to tell which phase actually failed.
  //
  // Observed error codes:
  //   container_nonzero_exit  → image pulled + container ran + exited != 0
  //                              (training/pip/start-script failure)
  //   disk_full              → disk-full mid-pull (failed during pulling)
  //   cancelled              → user cancelled (treated as cancellation, not error)
  //   timeout                → infrastructure timeout (provisioning level)
  //
  // For unknown codes, fall back to timestamp inference.
  const errorCode = job.error_code ?? null

  // Strongest "container is running" signal in priority order:
  //   1. started_running_at  (explicit, but Spark is slow to set this)
  //   2. container-phase log lines flowing  (we're literally seeing stdout
  //                                          from inside the container)
  //   3. error_code === 'container_nonzero_exit' (container ran then exited)
  //
  // hasLogs is true iff at least one log line has arrived in <LogStreamPanel>;
  // since Spark's agent banner ("Spark Compute Job") prints before the
  // container starts, hasLogs alone isn't enough — but combined with the
  // job no longer being just-queued, it's a reliable signal.
  const reachedRunning =
    !!runStart
    || hasLogs
    || (isFailed && errorCode === 'container_nonzero_exit')

  const failedAtPhase: 'queued' | 'provisioning' | 'pulling' | 'running' | null =
    !isFailed                                 ? null :
    errorCode === 'container_nonzero_exit'    ? 'running' :
    errorCode === 'disk_full'                 ? 'pulling' :
    errorCode === 'image-pull-failed'         ? 'pulling' :
    runStart                                  ? 'running' :
    heartbeat                                 ? 'pulling' :
    provStart                                 ? 'provisioning' :
                                                'queued'

  const out: Record<string, PhaseState> = {}

  // Walk forward; everything before `failedAtPhase` is done, the failed phase
  // gets `error`, everything after stays `pending`. For non-failed jobs we
  // use timestamp + log presence as before.
  const phaseDone = (k: string, completedAt?: string): PhaseState =>
    ({ status: 'done', completedAt })

  // Phase 1: queued
  if (failedAtPhase && failedAtPhase !== 'queued') {
    out.queued = phaseDone('queued', job.started_provisioning_at ?? job.created_at ?? undefined)
  } else if (failedAtPhase === 'queued') {
    out.queued = { status: 'error' }
  } else if (provStart) {
    out.queued = phaseDone('queued', job.started_provisioning_at ?? undefined)
  } else if (job.status === 'queued') {
    out.queued = { status: 'active', elapsedMs: created ? now - created : undefined }
  } else {
    out.queued = { status: 'pending' }
  }

  // Phase 2: provisioning EC2
  if (failedAtPhase && !['queued', 'provisioning'].includes(failedAtPhase)) {
    out.provisioning = phaseDone('provisioning',
      job.last_agent_heartbeat_at ?? job.started_running_at ?? job.terminal_at ?? undefined)
  } else if (failedAtPhase === 'provisioning') {
    out.provisioning = { status: 'error' }
  } else if (reachedRunning || heartbeat) {
    // hasLogs covers the case where Spark's job-record clock lags behind the
    // container actually running — if we're seeing stdout, provisioning is
    // long done, regardless of timestamps.
    out.provisioning = phaseDone('provisioning',
      job.last_agent_heartbeat_at ?? job.started_running_at ?? undefined)
  } else if (provStart && !terminal) {
    out.provisioning = { status: 'active', elapsedMs: now - provStart }
  } else {
    out.provisioning = { status: 'pending' }
  }

  // Phase 3: pulling image
  if (failedAtPhase === 'running') {
    out.pulling = phaseDone('pulling', job.started_running_at ?? undefined)
  } else if (failedAtPhase === 'pulling') {
    out.pulling = { status: 'error' }
  } else if (reachedRunning) {
    out.pulling = phaseDone('pulling', job.started_running_at ?? undefined)
  } else if (heartbeat && !terminal) {
    out.pulling = { status: 'active', elapsedMs: now - heartbeat }
  } else {
    out.pulling = { status: 'pending' }
  }

  // Phase 4: starting container — collapse with running. Done iff we have
  // strong evidence the container actually started.
  if (failedAtPhase === 'running' || reachedRunning) {
    out.starting = phaseDone('starting', job.started_running_at ?? undefined)
  } else {
    out.starting = { status: 'pending' }
  }

  // Phase 5: running
  if (failedAtPhase === 'running') {
    out.running = { status: 'error', completedAt: job.terminal_at ?? undefined }
  } else if (isTerminalSuccess && reachedRunning) {
    out.running = phaseDone('running', job.terminal_at ?? undefined)
  } else if (isCancelled && reachedRunning) {
    out.running = phaseDone('running', job.terminal_at ?? undefined)
  } else if (reachedRunning && !terminal) {
    // Active/running. Once we have logs the container is definitely up,
    // even if Spark's job.status hasn't transitioned out of "provisioning".
    out.running = { status: 'active', elapsedMs: runStart ? now - runStart : undefined }
  } else {
    out.running = { status: 'pending' }
  }

  return out
}

function formatRelative(iso: string): string {
  const ms = Date.now() - Date.parse(iso)
  if (Number.isNaN(ms) || ms < 0) return ''
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  return `${h}h ago`
}

function formatDur(ms: number): string {
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rs = s % 60
  return `${m}m ${rs}s`
}
