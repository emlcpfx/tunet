'use client'

/**
 * EZ-Comfy outputs panel — the comfy counterpart to the training Downloads
 * panel. A comfy job's deliverables are the rendered media files comfy_run.py
 * self-uploads to ShareSync (videos, stills) plus the converted
 * workflow_api.json, NOT checkpoints/exports. So this lists those directly:
 * inline <video>/<img> previews for the renders, a download button on each, and
 * a plain list for everything else (the workflow JSON, logs).
 *
 * Reuses the same files proxy as the training panel
 * (/api/spark/jobs/:id/files): no `download` flag streams the body inline (for
 * the previews), `&download=1` adds Content-Disposition for a real download.
 *
 * Polls the listing while empty (a comfy render finishes, then holds ~15s while
 * /output syncs to ShareSync), so the page fills in without a manual refresh.
 */

import { useEffect, useRef, useState } from 'react'
import type { SparkJob } from '@/lib/spark-types'

interface FileEntry {
  name:     string
  size:     number
  modified: string | null
}

const VIDEO_RE = /\.(mp4|webm|mov|m4v)$/i
const IMAGE_RE = /\.(png|jpe?g|webp|gif|avif)$/i

const POLL_MS    = 5_000
const POLL_TICKS = 36   // ~3 min of polling while outputs are still syncing

export function ComfyOutputsPanel({ job }: { job: SparkJob }) {
  const [files, setFiles]   = useState<FileEntry[] | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError]   = useState<string | null>(null)
  const [loading, setLoad]  = useState(true)
  const [reloadKey, setReloadKey] = useState(0)
  const ticksRef = useRef(0)

  // Manual refresh — also re-arms the auto-poll window (renders can outrun it,
  // and a failed render with no outputs must not poll forever).
  const refresh = () => { ticksRef.current = 0; setReloadKey(k => k + 1) }

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    async function load(initial: boolean) {
      if (initial) { setLoad(true); setError(null) }
      try {
        const res  = await fetch(`/api/spark/jobs/${job.id}/files`, { cache: 'no-store' })
        const data = await res.json()
        if (cancelled) return
        if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
        const entries = (data.entries ?? []) as FileEntry[]
        setFiles(entries)
        setPending(!!data.pending)
        // Keep polling while nothing has landed yet (render still running or
        // /output still syncing). Stop once files appear or we hit the cap.
        if (entries.length === 0 && ticksRef.current < POLL_TICKS) {
          ticksRef.current += 1
          timer = setTimeout(() => { void load(false) }, POLL_MS)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'failed')
      } finally {
        if (!cancelled && initial) setLoad(false)
      }
    }

    void load(true)
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [job.id, reloadKey])

  if (loading && !files) {
    return <Shell onRefresh={refresh}><p className="text-sm text-[#9ca3af]">Loading outputs…</p></Shell>
  }
  if (error && !files) {
    return (
      <Shell onRefresh={refresh}>
        <p className="text-sm text-[#EF4444]">Could not list output files: {error}</p>
        <p className="text-xs text-[#9ca3af] mt-1">
          The render may not have finished uploading to ShareSync yet, or SPARK_FILES_BASE_URL is not configured.
        </p>
      </Shell>
    )
  }
  if (!files || files.length === 0) {
    return (
      <Shell onRefresh={refresh}>
        <p className="text-sm text-[#9ca3af]">
          {pending
            ? 'Waiting for the render — outputs appear here once ComfyUI finishes and uploads /output to ShareSync.'
            : 'No output files yet. comfy_run.py uploads the render to ShareSync when it completes.'}
        </p>
      </Shell>
    )
  }

  const videos = files.filter(f => VIDEO_RE.test(f.name)).sort(byModifiedDesc)
  const images = files.filter(f => IMAGE_RE.test(f.name)).sort(byModifiedDesc)
  const others = files
    .filter(f => !VIDEO_RE.test(f.name) && !IMAGE_RE.test(f.name))
    .sort((a, b) => a.name.localeCompare(b.name))

  return (
    <Shell onRefresh={refresh}>
      <div className="space-y-4">
        {videos.map(f => (
          <Media key={f.name} jobId={job.id} file={f}>
            <video
              src={fileUrl(job.id, f.name)}
              controls
              preload="metadata"
              className="w-full rounded-md bg-black max-h-[60vh]"
            />
          </Media>
        ))}

        {images.map(f => (
          <Media key={f.name} jobId={job.id} file={f}>
            {/* eslint-disable-next-line @next/next/no-img-element -- streamed via our proxy, not a static asset */}
            <img
              src={fileUrl(job.id, f.name)}
              alt={f.name}
              className="w-full rounded-md bg-[#fafafa] object-contain max-h-[60vh]"
            />
          </Media>
        ))}

        {others.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-[#374151] mb-1">Other files</p>
            <ul className="divide-y divide-[#f3f4f6] border border-[#e5e7eb] rounded-md">
              {others.map(f => (
                <li key={f.name} className="flex items-center justify-between gap-3 px-3 py-2 text-xs">
                  <span className="font-mono text-[#374151] truncate">{f.name}</span>
                  <span className="flex items-center gap-3 flex-shrink-0">
                    <span className="text-[#9ca3af]">{formatBytes(f.size)}</span>
                    <a href={downloadUrl(job.id, f.name)} className="text-[#7E3AF2] hover:underline font-semibold">Download</a>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Shell>
  )
}

// ── Pieces ──────────────────────────────────────────────────────────────────

function Shell({ children, onRefresh }: { children: React.ReactNode; onRefresh?: () => void }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-[#111827]">Outputs</h3>
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            className="text-xs text-[#7E3AF2] hover:underline font-semibold"
          >
            ↻ Refresh
          </button>
        )}
      </div>
      {children}
    </div>
  )
}

/** A rendered media file: the preview (children) + a name/size/download row. */
function Media({ jobId, file, children }: { jobId: string; file: FileEntry; children: React.ReactNode }) {
  return (
    <div className="border border-[#e5e7eb] rounded-md overflow-hidden">
      {children}
      <div className="flex items-center justify-between gap-3 px-3 py-2 border-t border-[#e5e7eb]">
        <span className="font-mono text-xs text-[#374151] truncate" title={file.name}>{file.name}</span>
        <span className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-[#9ca3af]">{formatBytes(file.size)}</span>
          <a
            href={downloadUrl(jobId, file.name)}
            className="px-3 py-1.5 text-xs font-semibold bg-[#7E3AF2] hover:bg-[#6C2BD9] text-white rounded"
          >
            Download
          </a>
        </span>
      </div>
    </div>
  )
}

// ── helpers ──────────────────────────────────────────────────────────────────

function fileUrl(jobId: string, name: string): string {
  return `/api/spark/jobs/${jobId}/files?path=${encodeURIComponent(name)}`
}
function downloadUrl(jobId: string, name: string): string {
  return `${fileUrl(jobId, name)}&download=1`
}

function byModifiedDesc(a: FileEntry, b: FileEntry): number {
  const ta = a.modified ? Date.parse(a.modified) : 0
  const tb = b.modified ? Date.parse(b.modified) : 0
  return tb - ta
}

function formatBytes(n: number): string {
  if (!n) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}
