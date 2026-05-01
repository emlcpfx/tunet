/**
 * /demo/jobs/[id] — live job detail with SSE log stream.
 *
 * Server component for the surrounding chrome (job metadata fetched server-side
 * to avoid a client-side flash), client component for the SSE log viewer.
 *
 * Layout mirrors Spark/frontend/job-detail.html: header with status + GPU,
 * runtime metric strip, log panel that fills remaining height.
 */

import Link from 'next/link'
import { notFound } from 'next/navigation'
import { getJob } from '@/lib/spark'
import {
  type SparkJob, jobLabel, jobRuntimeMs, formatRuntime, formatStarted, ACTIVE_STATUSES,
} from '@/lib/spark-types'
import { SparkStatusBadge } from '@/components/spark/status-badge'
import { JobLiveView } from '@/components/spark/job-live-view'
import { CancelJobButton } from '@/components/spark/cancel-job-button'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export default async function JobDetailPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params

  let job: SparkJob | null = null
  try {
    job = await getJob(id)
  } catch (e) {
    return (
      <div className="px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
        Failed to load job: {e instanceof Error ? e.message : 'unknown'}
      </div>
    )
  }

  if (!job) notFound()

  const isLive = ACTIVE_STATUSES.has(job.status)

  return (
    <div className="space-y-5 animate-slide-in">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <Link href="/demo/jobs" className="text-sm text-[#6b7280] hover:text-[#374151]">
              ← All Jobs
            </Link>
          </div>
          <h1 className="text-2xl font-bold text-[#111827] font-mono">
            {jobLabel(job)}
          </h1>
          <p className="text-xs text-[#9ca3af] mt-1 font-mono break-all">{job.id}</p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <SparkStatusBadge status={job.status} />
          {isLive && <CancelJobButton jobId={job.id} />}
        </div>
      </div>

      {/* Metric strip */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <MetricBlock label="GPU"        value={job.gpu_name ?? '—'} />
        <MetricBlock label="Instance"   value={job.instance_type_name ?? '—'} mono />
        <MetricBlock label="Started"    value={formatStarted(job)} />
        <MetricBlock label="Runtime"    value={formatRuntime(jobRuntimeMs(job))} />
        <MetricBlock label="Image"      value={shortImage(job.image)} mono small />
      </div>

      {/* Error banner if failed */}
      {(job.status === 'failed' || (job.status === 'cancelled' && job.error_message)) && job.error_message && (
        <div className="px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm">
          <p className="font-semibold text-[#EF4444] mb-1">{job.error_code ?? 'Error'}</p>
          <p className="text-[#7F1D1D] font-mono text-xs">{job.error_message}</p>
        </div>
      )}

      {/* Provisioning timeline + log stream (combined client view) */}
      <section>
        <h2 className="text-sm font-semibold text-[#374151] mb-2 flex items-center gap-2">
          Live Output
          {isLive && (
            <span className="inline-flex items-center gap-1 text-xs text-[#16A34A] font-normal">
              <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] animate-pulse" />
              live
            </span>
          )}
        </h2>
        <JobLiveView initialJob={job} initiallyLive={isLive} />
      </section>

      {/* Output / ShareSync */}
      {job.output_share_sync_path && (
        <section>
          <h2 className="text-sm font-semibold text-[#374151] mb-2">Output</h2>
          <div className="bg-white border border-[#e5e7eb] rounded-lg px-4 py-3 text-xs font-mono text-[#374151] break-all">
            {job.output_share_sync_path}
          </div>
        </section>
      )}
    </div>
  )
}

function MetricBlock({ label, value, mono, small }: { label: string; value: string; mono?: boolean; small?: boolean }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg px-3 py-2.5">
      <p className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">{label}</p>
      <p className={`mt-1 text-[#111827] truncate ${mono ? 'font-mono' : ''} ${small ? 'text-xs' : 'text-sm'}`}>
        {value}
      </p>
    </div>
  )
}

function shortImage(img?: string): string {
  if (!img) return '—'
  // trim registry/long tags for readability
  const parts = img.split('/')
  const tail = parts[parts.length - 1]
  return tail.length > 30 ? tail.slice(0, 30) + '…' : tail
}
