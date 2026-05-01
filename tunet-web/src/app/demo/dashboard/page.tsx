/**
 * /demo/dashboard — live Spark Compute dashboard.
 *
 * Server component: fetches the job list directly from the Spark API on
 * each request (no DB, no Supabase). Revalidates on every load.
 *
 * Layout mirrors Spark/frontend/dashboard.html: metric strip + active job
 * card + recent jobs grid + "New Job" CTA.
 */

import Link from 'next/link'
import { listJobs } from '@/lib/spark'
import {
  type SparkJob, ACTIVE_STATUSES, jobLabel, jobRuntimeMs, formatRuntime,
} from '@/lib/spark-types'
import { SparkStatusBadge } from '@/components/spark/status-badge'
import { MetricCard } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export default async function DemoDashboard() {
  let jobs: SparkJob[] = []
  let fetchError: string | null = null

  try {
    jobs = await listJobs()
  } catch (e) {
    fetchError = e instanceof Error ? e.message : 'Failed to load jobs'
  }

  const active   = jobs.filter(j => ACTIVE_STATUSES.has(j.status))
  const recent   = jobs.slice(0, 8)
  const totalRun = jobs.length
  const completed = jobs.filter(j => j.status === 'completed').length

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#111827]">Dashboard</h1>
          <p className="text-sm text-[#6b7280] mt-1">
            Train image-to-image models on cloud GPUs. Pick a preset, point at frames, go.
          </p>
        </div>
        <Link href="/demo/jobs/new">
          <Button size="md">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
            New Job
          </Button>
        </Link>
      </div>

      {fetchError && (
        <div className="px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
          <strong>Couldn&apos;t load Spark jobs:</strong> {fetchError}
          <p className="text-xs mt-1 text-[#6b7280]">
            Check <code className="bg-white px-1 rounded">SPARK_EMAIL</code> / <code className="bg-white px-1 rounded">SPARK_PASSWORD</code> in <code className="bg-white px-1 rounded">.env.local</code>.
          </p>
        </div>
      )}

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
          <div className="bg-white border border-[#e5e7eb] rounded-xl overflow-hidden card-shadow">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#e5e7eb] bg-[#F9FAFB]">
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Job</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">GPU</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Status</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Started</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Duration</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody>
                {recent.map((j, i) => (
                  <tr key={j.id} className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} hover:bg-[#F9FAFB] transition-colors`}>
                    <td className="px-4 py-3">
                      <Link href={`/demo/jobs/${j.id}`} className="font-medium text-[#111827] hover:text-[#ae69f4] font-mono text-xs">
                        {jobLabel(j)}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-[#6b7280] text-xs">{j.instance_type_name ?? '—'}</td>
                    <td className="px-4 py-3"><SparkStatusBadge status={j.status} /></td>
                    <td className="px-4 py-3 text-[#6b7280] text-xs">{formatStartedRel(j)}</td>
                    <td className="px-4 py-3 text-[#6b7280] text-xs">
                      {formatRuntime(jobRuntimeMs(j))}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Link href={`/demo/jobs/${j.id}`} className="text-xs text-[#ae69f4] hover:underline">View</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
          <SparkStatusBadge status={job.status} />
          <div className="text-right">
            <p className="text-sm font-semibold text-[#111827]">{formatRuntime(jobRuntimeMs(job))}</p>
            <p className="text-xs text-[#9ca3af]">running</p>
          </div>
        </div>
      </div>
    </Link>
  )
}

/** "5 min ago", "2 hours ago", etc. */
function formatStartedRel(job: SparkJob): string {
  const ts = job.started_running_at ?? job.started_provisioning_at ?? job.created_at
  if (!ts) return '—'
  const diffMs = Date.now() - Date.parse(ts)
  if (Number.isNaN(diffMs) || diffMs < 0) return '—'
  const s = Math.floor(diffMs / 1000)
  if (s < 60)         return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60)         return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24)         return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}
