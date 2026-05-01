/**
 * /demo/jobs — full jobs list with filtering.
 *
 * Server component pulling live data from Spark on each request.
 * Filter by status via ?status=running etc.
 */

import Link from 'next/link'
import { listJobs } from '@/lib/spark'
import {
  type SparkJob, jobLabel, jobRuntimeMs, formatRuntime, formatStarted,
} from '@/lib/spark-types'
import { SparkStatusBadge } from '@/components/spark/status-badge'
import { Button } from '@/components/ui/button'

export const dynamic = 'force-dynamic'
export const revalidate = 0

const FILTERS: { label: string; value: string }[] = [
  { label: 'All',          value: 'all' },
  { label: 'Active',       value: 'active' },     // queued + provisioning + running
  { label: 'Running',      value: 'running' },
  { label: 'Completed',    value: 'completed' },
  { label: 'Failed',       value: 'failed' },
  { label: 'Cancelled',    value: 'cancelled' },
]

export default async function DemoJobsList({
  searchParams,
}: {
  searchParams: Promise<{ status?: string }>
}) {
  const params = await searchParams
  const statusFilter = params.status ?? 'all'

  let jobs: SparkJob[] = []
  let fetchError: string | null = null

  try {
    jobs = await listJobs()
  } catch (e) {
    fetchError = e instanceof Error ? e.message : 'Failed to load jobs'
  }

  const filtered = (() => {
    if (statusFilter === 'all')    return jobs
    if (statusFilter === 'active') return jobs.filter(j => ['queued','provisioning','running'].includes(j.status))
    return jobs.filter(j => j.status === statusFilter)
  })()

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#111827]">Training Jobs</h1>
          <p className="text-sm text-[#6b7280] mt-1">{jobs.length} total · live from Spark Compute v1</p>
        </div>
        <Link href="/demo/jobs/new">
          <Button>
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
        </div>
      )}

      {/* Filter tabs */}
      <div className="flex gap-1 bg-[#F9FAFB] border border-[#e5e7eb] rounded-lg p-1 w-fit overflow-x-auto">
        {FILTERS.map(f => (
          <Link
            key={f.value}
            href={`/demo/jobs?status=${f.value}`}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
              statusFilter === f.value
                ? 'bg-white text-[#111827] shadow-sm'
                : 'text-[#6b7280] hover:text-[#374151]'
            }`}
          >
            {f.label}
          </Link>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="bg-white border border-dashed border-[#D1D5DB] rounded-xl p-10 text-center">
          <p className="text-[#6b7280] text-sm mb-4">
            {statusFilter === 'all' ? 'No jobs yet' : `No ${statusFilter} jobs`}
          </p>
          {statusFilter === 'all' && (
            <Link href="/demo/jobs/new"><Button variant="secondary">Launch your first job</Button></Link>
          )}
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
              {filtered.map((j, i) => (
                <tr key={j.id} className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} hover:bg-[#F9FAFB] transition-colors`}>
                  <td className="px-4 py-3">
                    <Link href={`/demo/jobs/${j.id}`} className="font-mono text-xs text-[#111827] hover:text-[#ae69f4]">
                      {jobLabel(j)}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{j.instance_type_name ?? '—'}</td>
                  <td className="px-4 py-3"><SparkStatusBadge status={j.status} /></td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{formatStarted(j)}</td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{formatRuntime(jobRuntimeMs(j))}</td>
                  <td className="px-4 py-3 text-right">
                    <Link href={`/demo/jobs/${j.id}`} className="text-xs text-[#ae69f4] hover:underline">View →</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
