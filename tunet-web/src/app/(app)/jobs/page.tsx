/**
 * /jobs — full jobs list with filtering and bulk-cancel.
 *
 * Server component fetches the initial list directly from Spark; the client
 * component <JobsListLive> takes over for polling, filtering, multi-select,
 * and the bulk-cancel action.
 */

import Link from 'next/link'
import { listJobs } from '@/lib/spark'
import type { SparkJob } from '@/lib/spark-types'
import { JobsListLive } from '@/components/spark/jobs-list-live'
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

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#111827]">Training Jobs</h1>
          <p className="text-sm text-[#6b7280] mt-1">{jobs.length} total · live from Spark Compute v1</p>
        </div>
        <Link href="/jobs/new">
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
            href={`/jobs?status=${f.value}`}
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

      <JobsListLive initialJobs={jobs} statusFilter={statusFilter} />
    </div>
  )
}
