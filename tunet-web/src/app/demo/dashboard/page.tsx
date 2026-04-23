import Link from 'next/link'
import { MetricCard } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/badge'
import { formatCredits, formatDuration, type DbJob } from '@/types'

const MOCK_JOBS: DbJob[] = [
  {
    id: 'job-001',
    user_id: 'demo',
    name: 'dolly-shot-fix-v3',
    pod_id: 'ws-7821',
    status: 'running',
    gpu_type_id: 'spark_181',
    gpu_display_name: 'NVIDIA L40S 48GB',
    runpod_cost_per_hr: 3.0,
    platform_cost_per_hr: 4.50,
    accumulated_cost_cents: 135,
    billing_last_tick_at: new Date().toISOString(),
    config_path: 'jobs/job-001/config.yaml',
    src_zip_path: 'jobs/job-001/src.zip',
    dst_zip_path: 'jobs/job-001/dst.zip',
    checkpoint_path: null,
    container_disk_gb: 50,
    volume_gb: 100,
    started_at: new Date(Date.now() - 18 * 60 * 1000).toISOString(),
    ended_at: null,
    created_at: new Date(Date.now() - 20 * 60 * 1000).toISOString(),
  },
  {
    id: 'job-002',
    user_id: 'demo',
    name: 'hand-fix-v1',
    pod_id: 'ws-6644',
    status: 'terminated',
    gpu_type_id: 'spark_168',
    gpu_display_name: 'NVIDIA A10 24GB',
    runpod_cost_per_hr: 1.9,
    platform_cost_per_hr: 1.79,
    accumulated_cost_cents: 716,
    billing_last_tick_at: null,
    config_path: null,
    src_zip_path: null,
    dst_zip_path: null,
    checkpoint_path: null,
    container_disk_gb: 50,
    volume_gb: 100,
    started_at: new Date(Date.now() - 8 * 3600 * 1000).toISOString(),
    ended_at: new Date(Date.now() - 4 * 3600 * 1000).toISOString(),
    created_at: new Date(Date.now() - 8.1 * 3600 * 1000).toISOString(),
  },
  {
    id: 'job-003',
    user_id: 'demo',
    name: 'copycat-lora-test',
    pod_id: null,
    status: 'failed',
    gpu_type_id: 'spark_181',
    gpu_display_name: 'NVIDIA L40S 48GB',
    runpod_cost_per_hr: null,
    platform_cost_per_hr: 4.50,
    accumulated_cost_cents: 0,
    billing_last_tick_at: null,
    config_path: null,
    src_zip_path: null,
    dst_zip_path: null,
    checkpoint_path: null,
    container_disk_gb: 50,
    volume_gb: 100,
    started_at: null,
    ended_at: null,
    created_at: new Date(Date.now() - 24 * 3600 * 1000).toISOString(),
  },
]

const MOCK_BALANCE   = 5250   // cents
const MOCK_SPEND     = 851    // cents
const MOCK_GPU_HOURS = 4.7

export default function DemoDashboard() {
  const activeJobs = MOCK_JOBS.filter(j => j.status === 'running' || j.status === 'provisioning')

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#111827]">Dashboard</h1>
        <Link href="/demo/jobs/new">
          <Button size="md">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
            New Job
          </Button>
        </Link>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Credit Balance"   value={formatCredits(MOCK_BALANCE)}   sub="Available"       accent />
        <MetricCard label="Active Jobs"      value={String(activeJobs.length)}     sub="L40S running"           />
        <MetricCard label="Spend This Month" value={formatCredits(MOCK_SPEND)}     sub="Compute charges"        />
        <MetricCard label="GPU Hours"        value={MOCK_GPU_HOURS.toFixed(1)+'h'} sub="All time"               />
      </div>

      {activeJobs.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-[#111827] mb-3">Active Jobs</h2>
          <div className="space-y-3">
            {activeJobs.map(job => <ActiveJobCard key={job.id} job={job} />)}
          </div>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-[#111827]">Recent Jobs</h2>
          <Link href="/demo/jobs" className="text-sm text-[#ae69f4] hover:underline">View all</Link>
        </div>
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
              {MOCK_JOBS.map((job, i) => (
                <tr key={job.id} className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} hover:bg-[#F9FAFB] transition-colors`}>
                  <td className="px-4 py-3">
                    <Link href={`/demo/jobs/${job.id}`} className="font-medium text-[#111827] hover:text-[#ae69f4]">
                      {job.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-[#6b7280]">{job.gpu_display_name}</td>
                  <td className="px-4 py-3"><StatusBadge status={job.status} /></td>
                  <td className="px-4 py-3 text-[#6b7280]">
                    {job.started_at
                      ? formatDuration(((job.ended_at ? new Date(job.ended_at) : new Date()).getTime() - new Date(job.started_at).getTime()) / 1000)
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
      </div>
    </div>
  )
}

function ActiveJobCard({ job }: { job: DbJob }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl px-5 py-4 card-shadow flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-2 h-2 rounded-full bg-[#16A34A] animate-pulse flex-shrink-0" />
        <div className="min-w-0">
          <Link href={`/demo/jobs/${job.id}`} className="font-semibold text-[#111827] hover:text-[#ae69f4] truncate block">
            {job.name}
          </Link>
          <p className="text-xs text-[#6b7280]">
            {job.gpu_display_name} · {formatCredits((job.platform_cost_per_hr ?? 0) * 100)}/hr
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3 flex-shrink-0">
        <div className="text-right">
          <p className="text-sm font-semibold text-[#111827]">{formatCredits(job.accumulated_cost_cents)}</p>
          <p className="text-xs text-[#9ca3af]">spent</p>
        </div>
        <Link href={`/demo/jobs/${job.id}`}>
          <Button variant="secondary" size="sm">Monitor</Button>
        </Link>
      </div>
    </div>
  )
}
