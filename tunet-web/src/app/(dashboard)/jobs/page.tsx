import { auth } from '@clerk/nextjs/server'
import Link from 'next/link'
import { createServiceClient } from '@/lib/supabase'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/badge'
import { formatCredits, formatDuration, type DbJob, type JobStatus } from '@/types'

export const revalidate = 0

const STATUS_FILTERS: { label: string; value: JobStatus | 'all' }[] = [
  { label: 'All',        value: 'all' },
  { label: 'Running',    value: 'running' },
  { label: 'Stopped',    value: 'stopped' },
  { label: 'Terminated', value: 'terminated' },
  { label: 'Failed',     value: 'failed' },
]

export default async function JobsPage({
  searchParams,
}: {
  searchParams: Promise<{ status?: string; page?: string }>
}) {
  const { userId } = await auth()
  const params = await searchParams
  const statusFilter = (params.status ?? 'all') as JobStatus | 'all'
  const page = parseInt(params.page ?? '1', 10)
  const pageSize = 20

  const svc = createServiceClient()

  let query = svc
    .from('jobs')
    .select('*', { count: 'exact' })
    .eq('user_id', userId!)
    .order('created_at', { ascending: false })
    .range((page - 1) * pageSize, page * pageSize - 1)

  if (statusFilter !== 'all') {
    query = query.eq('status', statusFilter)
  }

  const { data: jobs, count } = await query
  const jobList = (jobs ?? []) as DbJob[]
  const totalPages = Math.ceil((count ?? 0) / pageSize)

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#111827]">Training Jobs</h1>
        <Link href="/jobs/new">
          <Button>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
            New Job
          </Button>
        </Link>
      </div>

      {/* Status filter tabs */}
      <div className="flex gap-1 bg-[#F9FAFB] border border-[#e5e7eb] rounded-lg p-1 w-fit">
        {STATUS_FILTERS.map(f => (
          <Link
            key={f.value}
            href={`/jobs?status=${f.value}`}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              statusFilter === f.value
                ? 'bg-white text-[#111827] shadow-sm'
                : 'text-[#6b7280] hover:text-[#374151]'
            }`}
          >
            {f.label}
          </Link>
        ))}
      </div>

      {/* Table */}
      {jobList.length === 0 ? (
        <div className="bg-white border border-dashed border-[#D1D5DB] rounded-xl p-10 text-center">
          <p className="text-[#6b7280] text-sm mb-4">
            {statusFilter === 'all' ? 'No jobs yet' : `No ${statusFilter} jobs`}
          </p>
          {statusFilter === 'all' && (
            <Link href="/jobs/new"><Button variant="secondary">Launch your first job</Button></Link>
          )}
        </div>
      ) : (
        <div className="bg-white border border-[#e5e7eb] rounded-xl overflow-hidden card-shadow">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#e5e7eb] bg-[#F9FAFB]">
                <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Job Name</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">GPU</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Status</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Started</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Duration</th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-[#374151]">Cost</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {jobList.map((job, i) => {
                const durationS = job.started_at
                  ? ((job.ended_at ? new Date(job.ended_at) : new Date()).getTime() -
                      new Date(job.started_at).getTime()) / 1000
                  : null
                return (
                  <tr
                    key={job.id}
                    className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} hover:bg-[#F9FAFB] transition-colors`}
                  >
                    <td className="px-4 py-3">
                      <Link href={`/jobs/${job.id}`} className="font-medium text-[#111827] hover:text-[#ae69f4]">
                        {job.name}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-[#6b7280] text-xs">
                      {job.gpu_display_name ?? job.gpu_type_id}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-4 py-3 text-[#6b7280] text-xs">
                      {job.started_at
                        ? new Date(job.started_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
                        : '—'}
                    </td>
                    <td className="px-4 py-3 text-[#6b7280] text-xs">
                      {durationS !== null ? formatDuration(durationS) : '—'}
                    </td>
                    <td className="px-4 py-3 text-right font-medium text-[#374151]">
                      {formatCredits(job.accumulated_cost_cents)}
                    </td>
                    <td className="px-4 py-3">
                      <Link href={`/jobs/${job.id}`}>
                        <button className="text-xs text-[#ae69f4] hover:underline">View</button>
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="border-t border-[#F3F4F6] px-4 py-3 flex items-center justify-between">
              <p className="text-xs text-[#6b7280]">Page {page} of {totalPages}</p>
              <div className="flex gap-2">
                {page > 1 && (
                  <Link href={`/jobs?status=${statusFilter}&page=${page - 1}`}>
                    <Button variant="ghost" size="sm">← Prev</Button>
                  </Link>
                )}
                {page < totalPages && (
                  <Link href={`/jobs?status=${statusFilter}&page=${page + 1}`}>
                    <Button variant="ghost" size="sm">Next →</Button>
                  </Link>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
