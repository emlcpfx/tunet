import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/components/ui/badge'
import { formatCredits, formatDuration, type DbJob } from '@/types'

const MOCK_JOBS: DbJob[] = [
  {
    id: 'job-001', user_id: 'demo', name: 'dolly-shot-fix-v3',
    pod_id: 'ws-7821', status: 'running',
    gpu_type_id: 'spark_181', gpu_display_name: 'NVIDIA L40S 48GB',
    runpod_cost_per_hr: 3.0, platform_cost_per_hr: 4.50,
    accumulated_cost_cents: 135, billing_last_tick_at: new Date().toISOString(),
    config_path: null, src_zip_path: null, dst_zip_path: null, checkpoint_path: null,
    container_disk_gb: 50, volume_gb: 100,
    started_at: new Date(Date.now() - 18 * 60 * 1000).toISOString(), ended_at: null,
    created_at: new Date(Date.now() - 20 * 60 * 1000).toISOString(),
  },
  {
    id: 'job-002', user_id: 'demo', name: 'hand-fix-v1',
    pod_id: 'ws-6644', status: 'terminated',
    gpu_type_id: 'spark_168', gpu_display_name: 'NVIDIA A10 24GB',
    runpod_cost_per_hr: 1.9, platform_cost_per_hr: 1.79,
    accumulated_cost_cents: 716, billing_last_tick_at: null,
    config_path: null, src_zip_path: null, dst_zip_path: null, checkpoint_path: null,
    container_disk_gb: 50, volume_gb: 100,
    started_at: new Date(Date.now() - 8 * 3600 * 1000).toISOString(),
    ended_at: new Date(Date.now() - 4 * 3600 * 1000).toISOString(),
    created_at: new Date(Date.now() - 8.1 * 3600 * 1000).toISOString(),
  },
  {
    id: 'job-003', user_id: 'demo', name: 'copycat-lora-test',
    pod_id: null, status: 'failed',
    gpu_type_id: 'spark_181', gpu_display_name: 'NVIDIA L40S 48GB',
    runpod_cost_per_hr: null, platform_cost_per_hr: 4.50,
    accumulated_cost_cents: 0, billing_last_tick_at: null,
    config_path: null, src_zip_path: null, dst_zip_path: null, checkpoint_path: null,
    container_disk_gb: 50, volume_gb: 100,
    started_at: null, ended_at: null,
    created_at: new Date(Date.now() - 24 * 3600 * 1000).toISOString(),
  },
  {
    id: 'job-004', user_id: 'demo', name: 'paintout-LoRA-v2',
    pod_id: 'ws-5512', status: 'terminated',
    gpu_type_id: 'spark_204', gpu_display_name: 'NVIDIA L40S Pro',
    runpod_cost_per_hr: 3.49, platform_cost_per_hr: 5.25,
    accumulated_cost_cents: 2100, billing_last_tick_at: null,
    config_path: null, src_zip_path: null, dst_zip_path: null, checkpoint_path: null,
    container_disk_gb: 50, volume_gb: 100,
    started_at: new Date(Date.now() - 3 * 24 * 3600 * 1000).toISOString(),
    ended_at: new Date(Date.now() - 3 * 24 * 3600 * 1000 + 4 * 3600 * 1000).toISOString(),
    created_at: new Date(Date.now() - 3 * 24 * 3600 * 1000 - 5 * 60 * 1000).toISOString(),
  },
]

export default function DemoJobsPage() {
  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#111827]">Training Jobs</h1>
        <Link href="/demo/jobs/new">
          <Button>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
            New Job
          </Button>
        </Link>
      </div>

      <div className="flex gap-1 bg-[#F9FAFB] border border-[#e5e7eb] rounded-lg p-1 w-fit">
        {(['All', 'Running', 'Stopped', 'Terminated', 'Failed'] as const).map(label => (
          <span
            key={label}
            className={`px-3 py-1.5 rounded-md text-sm font-medium ${
              label === 'All' ? 'bg-white text-[#111827] shadow-sm' : 'text-[#6b7280]'
            }`}
          >
            {label}
          </span>
        ))}
      </div>

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
            {MOCK_JOBS.map((job, i) => {
              const durationS = job.started_at
                ? ((job.ended_at ? new Date(job.ended_at) : new Date()).getTime() - new Date(job.started_at).getTime()) / 1000
                : null
              return (
                <tr key={job.id} className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''} hover:bg-[#F9FAFB] transition-colors`}>
                  <td className="px-4 py-3">
                    <Link href={`/demo/jobs/${job.id}`} className="font-medium text-[#111827] hover:text-[#ae69f4]">
                      {job.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-[#6b7280] text-xs">{job.gpu_display_name}</td>
                  <td className="px-4 py-3"><StatusBadge status={job.status} /></td>
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
                    <Link href={`/demo/jobs/${job.id}`}>
                      <button className="text-xs text-[#ae69f4] hover:underline">View</button>
                    </Link>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
