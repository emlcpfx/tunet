/**
 * /jobs/[id] — live job detail with SSE log stream.
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
  type SparkJob, jobLabel, jobRuntimeMs, formatRuntime, formatStarted, jobGpuDisplay,
  ACTIVE_STATUSES, derivedStatus,
} from '@/lib/spark-types'
import { LiveStatusBadge } from '@/components/spark/live-status-badge'
import { JobLiveView } from '@/components/spark/job-live-view'
import { CancelJobButton } from '@/components/spark/cancel-job-button'
import { SetMaxStepsButton } from '@/components/spark/set-max-steps-button'
import { TrainingChart } from '@/components/spark/training-chart'
import { TrainingStats } from '@/components/spark/training-stats'
import { PreviewImages } from '@/components/spark/preview-images'
import { DownloadsPanel } from '@/components/spark/downloads-panel'
import { JobSettingsPanel } from '@/components/spark/job-settings-panel'
import { ComfyOutputsPanel } from '@/components/comfy/comfy-outputs-panel'

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

  // Use derivedStatus() rather than raw job.status — Spark's job-level status
  // lags reality by minutes (it stays 'provisioning' even after the container
  // is running and emitting logs). derivedStatus checks heartbeat freshness
  // and started_running_at to surface a more accurate signal.
  // See spark-types.ts for the heuristic.
  const isLive = ACTIVE_STATUSES.has(derivedStatus(job))

  // EZ-Comfy jobs share this route but are a different beast — a one-shot
  // ComfyUI render, not a training run. They have no loss curve, no training
  // stats, and their deliverables are rendered media (not .pth/.onnx). Branch
  // the body to a comfy-specific layout. Marker is set by /api/comfy/submit.
  const isComfy = job.env?.TUNET_MODE === 'comfy'

  return (
    <div className="space-y-5 animate-slide-in">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3 mb-2">
            <Link href="/jobs" className="text-sm text-[#6b7280] hover:text-[#374151]">
              ← All Jobs
            </Link>
          </div>
          <h1 className="text-2xl font-bold text-[#111827] font-mono">
            {jobLabel(job)}
          </h1>
          <p className="text-xs text-[#9ca3af] mt-1 font-mono break-all">{job.id}</p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <LiveStatusBadge initialJob={job} />
          {/* Resume/Clone route into the training form — meaningless for a
              comfy render, so they're training-only. */}
          {!isComfy && !isLive && (
            <Link
              href={`/jobs/new?resume=${job.id}`}
              className="px-3 py-1.5 rounded-md text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] hover:bg-[#faf5ff] transition-colors"
              title="Continue training from this job's latest checkpoint"
            >
              Resume
            </Link>
          )}
          {isComfy ? (
            <Link
              href="/comfy"
              className="px-3 py-1.5 rounded-md text-xs font-semibold border border-[#e5e7eb] text-[#374151] hover:bg-[#F9FAFB] transition-colors"
              title="Start a new EZ-Comfy render"
            >
              New render
            </Link>
          ) : (
            <Link
              href={`/jobs/new?clone=${job.id}`}
              className="px-3 py-1.5 rounded-md text-xs font-semibold border border-[#e5e7eb] text-[#374151] hover:bg-[#F9FAFB] transition-colors"
              title="Start a new job pre-filled from this one"
            >
              Clone
            </Link>
          )}
          {isLive && !isComfy && <SetMaxStepsButton jobId={job.id} />}
          {isLive && <CancelJobButton jobId={job.id} />}
        </div>
      </div>

      {/* Metric strip */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <MetricBlock label="GPU"        value={jobGpuDisplay(job)} />
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

      {isComfy ? (
        /* Comfy render: a one-line settings summary + the rendered outputs
           (no training settings panel / chart / stats). */
        <>
          <div className="bg-white border border-[#e5e7eb] rounded-lg px-4 py-2.5 flex flex-wrap items-center gap-x-6 gap-y-1.5">
            <ComfyInfo label="Preset" value={job.env?.TUNET_PRESET ?? '—'} mono />
            <ComfyInfo label="GPU"    value={jobGpuDisplay(job)} />
            <ComfyInfo label="LoRA"   value={comfyLoraSummary(job.env)} mono />
          </div>
          <section>
            <ComfyOutputsPanel job={job} />
          </section>
        </>
      ) : (
        <>
          {/* Initial settings the job was submitted with */}
          <JobSettingsPanel job={job} />

          {/* Training chart + Preview images side-by-side */}
          <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <TrainingChart jobId={job.id} />
            <PreviewImages job={job} />
          </section>

          {/* Stats + analysis strip */}
          <section>
            <TrainingStats jobId={job.id} job={job} />
          </section>
        </>
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

      {/* Downloads (training only — comfy outputs are shown above) */}
      {!isComfy && job.output_share_sync_path && (
        <section>
          <h2 className="text-sm font-semibold text-[#374151] mb-2">Downloads</h2>
          <DownloadsPanel job={job} />
          <div className="mt-2 bg-white border border-[#e5e7eb] rounded-lg px-4 py-2 text-[11px] font-mono text-[#9ca3af] break-all">
            ShareSync path: {job.output_share_sync_path}
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

function ComfyInfo({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <span className="inline-flex items-baseline gap-2 min-w-0">
      <span className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">{label}</span>
      <span className={`text-sm text-[#111827] truncate ${mono ? 'font-mono' : ''}`} title={value}>{value}</span>
    </span>
  )
}

/**
 * One-line LoRA summary for a comfy job's settings strip. Prefers the stacked
 * COMFY_LORAS list (set when the preset supports a LoRA chain), falling back to
 * the preset's single default (COMFY_LORA_NAME). See lib/comfy.ts buildComfyEnv.
 */
function comfyLoraSummary(env: SparkJob['env']): string {
  if (!env) return '—'
  const rawList = env.COMFY_LORAS
  if (rawList) {
    try {
      const arr = JSON.parse(rawList) as { file?: string; strength?: number }[]
      if (Array.isArray(arr) && arr.length > 0) {
        return arr
          .map(l => {
            const name = (l.file ?? '').split('/').pop() || (l.file ?? '?')
            return typeof l.strength === 'number' ? `${name} (${l.strength})` : name
          })
          .join(', ')
      }
    } catch { /* fall through to the single-LoRA default */ }
  }
  return env.COMFY_LORA_NAME || 'none'
}

function shortImage(img?: string): string {
  if (!img) return '—'
  // trim registry/long tags for readability
  const parts = img.split('/')
  const tail = parts[parts.length - 1]
  return tail.length > 30 ? tail.slice(0, 30) + '…' : tail
}
