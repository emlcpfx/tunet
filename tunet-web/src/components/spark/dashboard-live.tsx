'use client'

/**
 * Client component that owns the live job state for /demo/dashboard.
 *
 * Seeded from the server-fetched job list to avoid first-paint flash, then
 * polls /api/spark/jobs every 5s so status, runtimes, and the active card
 * update without a hard reload.
 *
 * Renders: metric strip, active job card(s), recent jobs table.
 */

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import {
  type SparkJob, ACTIVE_STATUSES, jobLabel, jobRuntimeMs, formatRuntime, derivedStatus,
} from '@/lib/spark-types'
import { SparkStatusBadge } from '@/components/spark/status-badge'
import { JobsTable } from '@/components/spark/jobs-table'
import { MetricCard } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface DashboardLiveProps {
  initialJobs: SparkJob[]
}

export function DashboardLive({ initialJobs }: DashboardLiveProps) {
  const [jobs, setJobs] = useState<SparkJob[]>(initialJobs)
  const [pollError, setPollError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/spark/jobs', { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      if (Array.isArray(data.jobs)) setJobs(data.jobs as SparkJob[])
      setPollError(null)
    } catch (e) {
      setPollError(e instanceof Error ? e.message : 'poll failed')
    }
  }, [])

  // Poll job list. Fast cadence when something is active, slow when idle so we
  // don't hammer Spark when the user has the tab open in the background.
  useEffect(() => {
    const anyActive = jobs.some(j => ACTIVE_STATUSES.has(j.status))
    const interval = anyActive ? 5_000 : 20_000
    const id = window.setInterval(refresh, interval)
    return () => window.clearInterval(id)
  }, [jobs, refresh])

  // Tick clock so runtime/relative-time labels update between polls.
  useEffect(() => {
    const id = window.setInterval(() => setTick(t => t + 1), 1000)
    return () => window.clearInterval(id)
  }, [])

  const active    = jobs.filter(j => ACTIVE_STATUSES.has(j.status))
  const recent    = jobs.slice(0, 8)
  const totalRun  = jobs.length
  const completed = jobs.filter(j => j.status === 'completed').length

  return (
    <div className="space-y-6" data-tick={tick}>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          label="Active Jobs"
          value={String(active.length)}
          sub={active.length === 0 ? 'No jobs running' : `${active.length} training`}
          accent={active.length > 0}
        />
        <MetricCard
          label="Completed"
          value={String(completed)}
          sub={`of ${totalRun} total`}
        />
        <MetricCard
          label="Last Activity"
          value={recent[0] ? formatStartedRel(recent[0]) : '—'}
          sub={recent[0] ? recent[0].instance_type_name ?? '?' : 'No jobs yet'}
        />
      </div>

      {pollError && (
        <div className="px-3 py-2 bg-[#FFFBEB] border border-[#fde68a] rounded text-xs text-[#92400E]">
          Refresh failed: {pollError}
        </div>
      )}

      {active.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-[#111827] mb-3">Active</h2>
          <div className="space-y-3">
            {active.map(j => <ActiveJobCard key={j.id} job={j} />)}
          </div>
        </section>
      )}

      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-[#111827]">Recent Jobs</h2>
          <Link href="/demo/jobs" className="text-sm text-[#ae69f4] hover:underline">View all →</Link>
        </div>

        {recent.length === 0 ? (
          <div className="bg-white border border-dashed border-[#D1D5DB] rounded-xl p-10 text-center">
            <p className="text-[#6b7280] text-sm mb-4">No jobs yet</p>
            <Link href="/demo/jobs/new"><Button variant="secondary">Launch your first job</Button></Link>
          </div>
        ) : (
          <JobsTable jobs={recent} startedFormat={formatStartedRel} onAfterAction={refresh} />
        )}
      </section>
    </div>
  )
}

function ActiveJobCard({ job }: { job: SparkJob }) {
  return (
    <Link
      href={`/demo/jobs/${job.id}`}
      className="block bg-white border border-[#e5e7eb] rounded-xl px-5 py-4 card-shadow hover:border-[#ae69f4] transition-colors"
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-2 h-2 rounded-full bg-[#16A34A] animate-pulse flex-shrink-0" />
          <div className="min-w-0">
            <p className="font-semibold text-[#111827] truncate font-mono text-sm">{jobLabel(job)}</p>
            <p className="text-xs text-[#6b7280] mt-0.5">
              {job.instance_type_name ?? 'unknown SKU'} · {job.gpu_name ?? '—'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4 flex-shrink-0">
          <SparkStatusBadge status={job.status} liveOverride={derivedStatus(job)} />
          <div className="text-right">
            <p className="text-sm font-semibold text-[#111827]">{formatRuntime(jobRuntimeMs(job))}</p>
            <p className="text-xs text-[#9ca3af]">running</p>
          </div>
        </div>
      </div>
    </Link>
  )
}

function formatStartedRel(job: SparkJob): string {
  const ts = job.started_running_at ?? job.started_provisioning_at ?? job.created_at
  if (!ts) return '—'
  const diffMs = Date.now() - Date.parse(ts)
  if (Number.isNaN(diffMs) || diffMs < 0) return '—'
  const s = Math.floor(diffMs / 1000)
  if (s < 60)  return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60)  return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24)  return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}
