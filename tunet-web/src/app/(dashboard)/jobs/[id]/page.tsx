'use client'
import { useEffect, useState, useCallback, use } from 'react'
import { useRouter } from 'next/navigation'
import Image from 'next/image'
import { Card, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/badge'
import { TrainingChart } from '@/components/dashboard/training-chart'
import { LogViewer } from '@/components/dashboard/log-viewer'
import { formatCredits, formatDuration, type DbJob, type MonitorStatus, type MonitorMetrics, type MonitorLogs } from '@/types'

const POLL_INTERVAL_MS = 10_000
const PREVIEW_INTERVAL_MS = 30_000

export default function JobDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()

  const [job, setJob]           = useState<DbJob | null>(null)
  const [status, setStatus]     = useState<MonitorStatus | null>(null)
  const [metrics, setMetrics]   = useState<MonitorMetrics | null>(null)
  const [logs, setLogs]         = useState<string[]>([])
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loadingJob, setLoadingJob] = useState(true)
  const [actionLoading, setActionLoading] = useState<'stop' | 'terminate' | null>(null)
  const [monitorError, setMonitorError] = useState<string | null>(null)

  // Fetch job record
  const fetchJob = useCallback(async () => {
    const res = await fetch(`/api/jobs/${id}`)
    if (res.ok) {
      const data = await res.json() as DbJob
      setJob(data)
    }
    setLoadingJob(false)
  }, [id])

  // Poll monitor_api
  const pollMonitor = useCallback(async () => {
    if (!job?.pod_id || job.status !== 'running') return

    const [statusRes, metricsRes, logsRes] = await Promise.all([
      fetch(`/api/jobs/${id}/monitor?path=/api/status`),
      fetch(`/api/jobs/${id}/monitor?path=/api/metrics`),
      fetch(`/api/jobs/${id}/monitor?path=/api/logs`),
    ])

    if (statusRes.ok) {
      const s = await statusRes.json() as { data: MonitorStatus; error: null } | { data: null; error: string }
      if (s.data) { setStatus(s.data); setMonitorError(null) }
      else setMonitorError(s.error)
    }
    if (metricsRes.ok) {
      const m = await metricsRes.json() as { data: MonitorMetrics }
      if (m.data) setMetrics(m.data)
    }
    if (logsRes.ok) {
      const l = await logsRes.json() as { data: MonitorLogs }
      if (l.data) setLogs(l.data.lines)
    }
  }, [id, job?.pod_id, job?.status])

  // Poll preview image URL
  const pollPreview = useCallback(async () => {
    if (!job?.pod_id || job.status !== 'running') return
    const res = await fetch(`/api/jobs/${id}/monitor?path=/api/preview`, { method: 'HEAD' })
    if (res.ok) {
      // Use the proxy endpoint as image src
      setPreviewUrl(`/api/jobs/${id}/monitor?path=/api/preview&t=${Date.now()}`)
    }
  }, [id, job?.pod_id, job?.status])

  useEffect(() => { fetchJob() }, [fetchJob])

  useEffect(() => {
    const t = setInterval(() => { fetchJob(); pollMonitor() }, POLL_INTERVAL_MS)
    return () => clearInterval(t)
  }, [fetchJob, pollMonitor])

  useEffect(() => {
    const t = setInterval(pollPreview, PREVIEW_INTERVAL_MS)
    pollPreview()
    return () => clearInterval(t)
  }, [pollPreview])

  // Initial monitor poll
  useEffect(() => { pollMonitor() }, [pollMonitor])

  async function handleStop() {
    setActionLoading('stop')
    await fetch(`/api/jobs/${id}`, { method: 'PATCH', body: JSON.stringify({ action: 'stop' }), headers: { 'Content-Type': 'application/json' } })
    await fetchJob()
    setActionLoading(null)
  }

  async function handleTerminate() {
    if (!confirm('Terminate this pod? Billing stops immediately. Make sure you\'ve downloaded your checkpoints.')) return
    setActionLoading('terminate')
    await fetch(`/api/jobs/${id}`, { method: 'PATCH', body: JSON.stringify({ action: 'terminate' }), headers: { 'Content-Type': 'application/json' } })
    await fetchJob()
    setActionLoading(null)
  }

  if (loadingJob) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-[#ae69f4] border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!job) {
    return (
      <div className="text-center py-20">
        <p className="text-[#6b7280]">Job not found</p>
        <Button className="mt-4" variant="ghost" onClick={() => router.push('/jobs')}>Back to Jobs</Button>
      </div>
    )
  }

  const isRunning = job.status === 'running'
  const elapsed = job.started_at
    ? ((job.ended_at ? new Date(job.ended_at) : new Date()).getTime() - new Date(job.started_at).getTime()) / 1000
    : null

  return (
    <div className="space-y-5 animate-slide-in">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold text-[#111827]">{job.name}</h1>
            <StatusBadge status={job.status} />
          </div>
          <p className="text-sm text-[#6b7280] mt-1">
            {job.gpu_display_name ?? job.gpu_type_id}
            {job.platform_cost_per_hr && ` · ${formatCredits(Math.round(job.platform_cost_per_hr * 100))}/hr`}
            {elapsed !== null && ` · ${formatDuration(elapsed)}`}
          </p>
        </div>
        <div className="flex gap-2 flex-shrink-0">
          {isRunning && (
            <Button
              variant="secondary"
              size="sm"
              loading={actionLoading === 'stop'}
              onClick={handleStop}
            >
              Stop
            </Button>
          )}
          {(isRunning || job.status === 'stopped') && (
            <Button
              variant="danger"
              size="sm"
              loading={actionLoading === 'terminate'}
              onClick={handleTerminate}
            >
              Terminate
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={() => router.push('/jobs')}>← Back</Button>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Cost" value={formatCredits(job.accumulated_cost_cents)} />
        <StatCard label="Step" value={status?.step !== undefined ? String(status.step) : '—'} />
        <StatCard
          label={`${status?.loss_label ?? 'Loss'}`}
          value={status?.loss !== null && status?.loss !== undefined ? status.loss.toFixed(5) : '—'}
        />
        <StatCard
          label="Step Time"
          value={status?.step_time_s !== null && status?.step_time_s !== undefined ? `${status.step_time_s.toFixed(2)}s` : '—'}
        />
      </div>

      {/* Monitor error */}
      {monitorError && isRunning && (
        <div className="p-3 bg-[#FFFBEB] border border-[#FDE68A] rounded-lg text-xs text-[#D97706]">
          Monitor: {monitorError} — Pod may still be starting up
        </div>
      )}

      {/* Loss chart */}
      <Card>
        <CardTitle className="mb-4">Loss Chart</CardTitle>
        <TrainingChart metrics={metrics} />
      </Card>

      {/* Preview + Logs side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Training preview */}
        <Card>
          <CardTitle className="mb-3">Training Preview</CardTitle>
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Training preview"
              className="w-full rounded-lg object-contain max-h-64"
            />
          ) : (
            <div className="flex items-center justify-center h-40 bg-[#F9FAFB] rounded-lg text-sm text-[#9ca3af]">
              {isRunning ? 'Waiting for first preview...' : 'No preview available'}
            </div>
          )}
        </Card>

        {/* Logs */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <CardTitle>Training Logs</CardTitle>
            {isRunning && (
              <span className="text-xs text-[#16A34A] flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] animate-pulse" />
                Live
              </span>
            )}
          </div>
          <LogViewer lines={logs} height="220px" />
        </Card>
      </div>

      {/* Job metadata */}
      <Card>
        <CardTitle className="mb-3">Job Details</CardTitle>
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-3 text-sm">
          <DetailRow label="Pod ID"    value={job.pod_id ?? '—'} mono />
          <DetailRow label="Job ID"    value={job.id} mono />
          <DetailRow label="GPU"       value={job.gpu_display_name ?? job.gpu_type_id} />
          <DetailRow label="Container" value={`${job.container_disk_gb} GB disk`} />
          <DetailRow label="Volume"    value={`${job.volume_gb} GB persistent`} />
          <DetailRow label="Created"   value={new Date(job.created_at).toLocaleString()} />
        </dl>
      </Card>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl px-4 py-3 card-shadow">
      <p className="text-xs text-[#6b7280]">{label}</p>
      <p className="text-lg font-bold text-[#111827] mt-0.5">{value}</p>
    </div>
  )
}

function DetailRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-xs text-[#9ca3af] mb-0.5">{label}</dt>
      <dd className={`text-[#374151] truncate ${mono ? 'font-mono text-xs' : ''}`}>{value}</dd>
    </div>
  )
}
