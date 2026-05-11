import { auth } from '@/auth'
import Link from 'next/link'
import { createServiceClient } from '@/lib/supabase'
import { MetricCard } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/badge'
import { formatCredits, formatDuration, type DbJob, type DbBillingEvent } from '@/types'

export const revalidate = 0

export default async function DashboardPage() {
  const session = await auth()
  const userId = session!.user!.id
  const svc = createServiceClient()

  const [{ data: user }, { data: jobs }, { data: events }] = await Promise.all([
    svc.from('users').select('*').eq('id', userId!).single(),
    svc
      .from('jobs')
      .select('*')
      .eq('user_id', userId!)
      .order('created_at', { ascending: false })
      .limit(10),
    svc
      .from('billing_events')
      .select('amount_cents')
      .eq('user_id', userId!)
      .eq('type', 'compute_charge')
      .gte('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString()),
  ])

  const jobList = (jobs ?? []) as DbJob[]
  const activeJobs  = jobList.filter(j => j.status === 'running' || j.status === 'provisioning')
  const monthlySpend = (events ?? [] as DbBillingEvent[]).reduce(
    (sum, e) => sum + Math.abs(e.amount_cents), 0,
  )
  const gpuHours = jobList
    .filter(j => j.started_at && j.ended_at)
    .reduce((sum, j) => {
      const ms = new Date(j.ended_at!).getTime() - new Date(j.started_at!).getTime()
      return sum + ms / 3_600_000
    }, 0)

  return (
    <div className="space-y-6 animate-slide-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#111827]">Dashboard</h1>
        <Link href="/jobs/new">
          <Button size="md">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
            New Job
          </Button>
        </Link>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Credit Balance"
          value={formatCredits(user?.credit_balance_cents ?? 0)}
          sub="Available"
          accent
        />
        <MetricCard
          label="Active Jobs"
          value={String(activeJobs.length)}
          sub={activeJobs.length > 0 ? `${activeJobs.map(j => j.gpu_display_name ?? j.gpu_type_id).join(', ')}` : 'None running'}
        />
        <MetricCard
          label="Spend This Month"
          value={formatCredits(monthlySpend)}
          sub="Compute charges"
        />
        <MetricCard
          label="GPU Hours"
          value={gpuHours.toFixed(1) + 'h'}
          sub="All time"
        />
      </div>

      {/* Active jobs */}
      {activeJobs.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-[#111827] mb-3">Active Jobs</h2>
          <div className="space-y-3">
            {activeJobs.map(job => (
              <ActiveJobCard key={job.id} job={job} />
            ))}
          </div>
        </div>
      )}

      {/* Recent jobs */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-[#111827]">Recent Jobs</h2>
          <Link href="/jobs" className="text-sm text-[#ae69f4] hover:underline">View all</Link>
        </div>

        {jobList.length === 0 ? (
          <div className="bg-white border border-dashed border-[#D1D5DB] rounded-xl p-10 text-center">
            <p className="text-[#6b7280] text-sm mb-4">No training jobs yet</p>
            <Link href="/jobs/new">
              <Button variant="secondary">Launch your first job</Button>
            </Link>
          </div>
        ) : (
          <div className="bg-white border border-[#e5e7eb] rounded-xl overflow-hidden card-shadow">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#e5e7eb] bg-[#F9FAFB]">
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Job</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">GPU</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Status</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-[#374151]">Duration</th>
                  <th className="text-right px-4 py-3 text-xs font-semibold text-[#374151]">Cost</th>
                </tr>
              </thead>
              <tbody>
                {jobList.map((job, i) => (
                  <tr
                    key={job.id}
                    className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} hover:bg-[#F9FAFB] transition-colors`}
                  >
                    <td className="px-4 py-3">
                      <Link href={`/jobs/${job.id}`} className="font-medium text-[#111827] hover:text-[#ae69f4]">
                        {job.name}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-[#6b7280]">
                      {job.gpu_display_name ?? job.gpu_type_id}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-4 py-3 text-[#6b7280]">
                      {job.started_at
                        ? formatDuration(
                            ((job.ended_at ? new Date(job.ended_at) : new Date()).getTime() -
                              new Date(job.started_at).getTime()) / 1000,
                          )
                        : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-[#374151]">
                      {formatCredits(job.accumulated_cost_cents)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function ActiveJobCard({ job }: { job: DbJob }) {
  const ratePerHr = job.platform_cost_per_hr ?? 0
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl px-5 py-4 card-shadow flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-2 h-2 rounded-full bg-[#16A34A] animate-pulse flex-shrink-0" />
        <div className="min-w-0">
          <Link href={`/jobs/${job.id}`} className="font-semibold text-[#111827] hover:text-[#ae69f4] truncate block">
            {job.name}
          </Link>
          <p className="text-xs text-[#6b7280]">
            {job.gpu_display_name ?? job.gpu_type_id} · {formatCredits(ratePerHr * 100)}/hr
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3 flex-shrink-0">
        <div className="text-right">
          <p className="text-sm font-semibold text-[#111827]">{formatCredits(job.accumulated_cost_cents)}</p>
          <p className="text-xs text-[#9ca3af]">spent</p>
        </div>
        <Link href={`/jobs/${job.id}`}>
          <Button variant="secondary" size="sm">Monitor</Button>
        </Link>
      </div>
    </div>
  )
}
