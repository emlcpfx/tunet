'use client'

/**
 * Downloads panel — lists output files (checkpoints + ONNX exports) from
 * ShareSync and offers download buttons. Mirrors the four canonical exports
 * from gui/__init__.py (Latest .pth for fine-tune, Flame/AE ONNX, Nuke ONNX,
 * Preview JPGs). Lists per-epoch .pth files in a collapsed sub-list.
 */

import { useEffect, useState } from 'react'
import type { SparkJob } from '@/lib/spark-types'

interface FileEntry {
  name:     string
  size:     number
  modified: string | null
}

interface DownloadsPanelProps {
  job: SparkJob
}

export function DownloadsPanel({ job }: DownloadsPanelProps) {
  const [files, setFiles]   = useState<FileEntry[] | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError]   = useState<string | null>(null)
  const [loading, setLoad]  = useState(true)
  const [expanded, setExp]  = useState(false)
  const [exportState, setExportState] = useState<
    | { kind: 'idle' }
    | { kind: 'submitting' }
    | { kind: 'submitted'; jobId: string; checkpoint: string }
    | { kind: 'ready';     downloadUrl: string; checkpoint: string; exportName: string }
    | { kind: 'error';     msg: string }
  >({ kind: 'idle' })

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoad(true); setError(null)
      try {
        const res = await fetch(`/api/spark/jobs/${job.id}/files`, { cache: 'no-store' })
        const data = await res.json()
        if (cancelled) return
        if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
        setFiles((data.entries ?? []) as FileEntry[])
        setPending(!!data.pending)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'failed')
      } finally {
        if (!cancelled) setLoad(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [job.id])

  if (loading && !files) {
    return (
      <div className="bg-white border border-[#e5e7eb] rounded-lg p-4 text-sm text-[#9ca3af]">
        Loading downloads…
      </div>
    )
  }
  if (error && !files) {
    return (
      <div className="bg-white border border-[#e5e7eb] rounded-lg p-4 text-sm">
        <span className="text-[#EF4444]">Could not list output files: {error}</span>
        <p className="text-xs text-[#9ca3af] mt-1">
          The job&apos;s output may not have been uploaded to ShareSync yet, or
          SPARK_FILES_BASE_URL is not configured.
        </p>
      </div>
    )
  }
  if (!files || files.length === 0) {
    return (
      <div className="bg-white border border-[#e5e7eb] rounded-lg p-4 text-sm text-[#9ca3af]">
        {pending
          ? 'Waiting for first file — output dir will appear in ShareSync once training writes its first checkpoint.'
          : 'No output files yet. The agent uploads /output/ as files become quiet (~3-5s).'}
      </div>
    )
  }

  // Classify. Exports land under exports/<flame|nuke>/… (the files route
  // descends one level into them), so classify by that subfolder rather than
  // the basename — and surface the most-recent when several epochs exist.
  const flameOnnx = files.filter(f => /(^|\/)flame\//i.test(f.name) && f.name.endsWith('.onnx'))
                         .sort(byModifiedDesc)[0] ?? null
  // Nuke's deliverable is a trio — .cat (Cattery model) + .nk (node) + .pt
  // (TorchScript) — NOT an .onnx. Group them and surface the .cat as primary.
  const nukeFiles = files.filter(f => /(^|\/)nuke\//i.test(f.name) && /\.(cat|nk|pt)$/i.test(f.name))
                         .sort(byModifiedDesc)
  const nukePrimary = nukeFiles.find(f => f.name.endsWith('.cat')) ?? nukeFiles[0] ?? null
  const allExports  = files.filter(f => f.name.startsWith('exports/')).sort(byModifiedDesc)
  const pthLatest = files.find(f => /_latest\.pth$/.test(f.name))
                ?? files.filter(f => f.name.endsWith('.pth')).sort(byModifiedDesc)[0]
  const pthEpochs = files.filter(f => f.name.endsWith('.pth') && f.name !== pthLatest?.name)
                         .sort(byModifiedDesc)
  const previewJpgs = files.filter(f => /preview/i.test(f.name) && /\.jpe?g$/i.test(f.name))
  const trainingLog = files.find(f => f.name === 'training.log')
  const otherLogs   = files.filter(f => f.name.endsWith('.log') && f.name !== 'training.log')

  // "Export now" handler — POSTs to /api/spark/jobs/:id/export-onnx.
  // Two possible success shapes:
  //   { fromCache: true,  downloadUrl, … }  → auto-export already produced
  //     a fresh .onnx; we surface a direct download link, no new compute.
  //   { fromCache: false, jobId, … }        → no fresh export available;
  //     a tiny Spark job was spawned, link to its progress page.
  async function handleExport() {
    setExportState({ kind: 'submitting' })
    try {
      const res = await fetch(`/api/spark/jobs/${job.id}/export-onnx`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({}),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
      if (data.fromCache) {
        setExportState({
          kind:        'ready',
          downloadUrl: data.downloadUrl,
          checkpoint:  data.checkpoint,
          exportName:  data.exportName,
        })
      } else {
        setExportState({
          kind:       'submitted',
          jobId:      data.jobId,
          checkpoint: data.checkpoint,
        })
      }
    } catch (e) {
      setExportState({ kind: 'error', msg: e instanceof Error ? e.message : 'submit failed' })
    }
  }

  // The export-onnx route needs a real .pth to read from ShareSync. Until
  // the first epoch lands the button is dead weight, so disable it.
  const canExport = !!pthLatest

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg p-4">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">Downloads</h3>
      <p className="text-xs text-[#6b7280] mb-3">
        Latest checkpoint and exports. <span className="font-semibold">Export now</span> grabs the
        most recent auto-exported ONNX from training (instant download) — or if none exists yet, falls
        back to a tiny CPU job (~2 min, ~$0.02) that converts on demand.
      </p>

      {/* ── Export now action row ─────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-3 p-2 rounded-md bg-[#faf5ff] border border-[#e9d5ff]">
        <button
          onClick={handleExport}
          disabled={
            !canExport ||
            exportState.kind === 'submitting' ||
            exportState.kind === 'submitted' ||
            exportState.kind === 'ready'
          }
          className={`text-xs font-semibold px-3 py-1.5 rounded-md transition-colors ${
            !canExport || exportState.kind === 'submitting' || exportState.kind === 'submitted' || exportState.kind === 'ready'
              ? 'bg-[#e5e7eb] text-[#9ca3af] cursor-not-allowed'
              : 'bg-[#7E3AF2] text-white hover:bg-[#6D28D9] cursor-pointer'
          }`}
        >
          {exportState.kind === 'submitting' ? 'Checking…' : 'Export now'}
        </button>
        <div className="text-xs text-[#6b7280] min-w-0 flex-1">
          {exportState.kind === 'idle' && (canExport
            ? <>Convert <span className="font-mono text-[#374151]">{pthLatest?.name}</span> → ONNX + TorchScript</>
            : 'Waiting for the first checkpoint…')}
          {exportState.kind === 'submitting' && 'Looking for a ready export…'}
          {exportState.kind === 'ready' && (
            <>
              Ready from training:{' '}
              <a
                href={exportState.downloadUrl}
                download={exportState.exportName}
                className="text-[#7E3AF2] hover:underline font-semibold"
              >
                Download {exportState.exportName} ↓
              </a>
            </>
          )}
          {exportState.kind === 'submitted' && (
            <>
              No fresh auto-export — spawned a job from <span className="font-mono text-[#374151]">{exportState.checkpoint}</span>.{' '}
              <a
                href={`/demo/jobs/${exportState.jobId}`}
                className="text-[#7E3AF2] hover:underline font-semibold"
              >
                View progress →
              </a>{' '}
              Files will appear here when it completes.
            </>
          )}
          {exportState.kind === 'error' && (
            <span className="text-[#EF4444]">Failed: {exportState.msg}</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <DownloadCard
          jobId={job.id}
          file={pthLatest}
          title="Latest checkpoint (.pth)"
          subtitle="For PyTorch resume / fine-tuning"
          accent
        />
        <DownloadCard
          jobId={job.id}
          file={flameOnnx}
          title="Export for Flame / After Effects"
          subtitle="No export yet — use Export now"
        />
        <DownloadCard
          jobId={job.id}
          file={nukePrimary}
          title="Export for Nuke"
          subtitle="No export yet — use Export now"
          extraDownloadFiles={nukeFiles.filter(f => f !== nukePrimary)}
        />
        <DownloadCard
          jobId={job.id}
          file={previewJpgs[0] ?? null}
          title="Preview images"
          subtitle={previewJpgs.length > 0 ? `${previewJpgs.length} files (download all from list below)` : 'Not generated yet'}
          extraDownloadFiles={previewJpgs.length > 1 ? previewJpgs.slice(1) : undefined}
        />
      </div>

      {/* Every file in the exports/ tree — Flame .onnx/.json, Nuke .cat/.nk/.pt,
          across all exported epochs — each individually downloadable. The two
          cards above are just quick links to the latest of each. */}
      {allExports.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-semibold text-[#374151] mb-1">All exports ({allExports.length})</p>
          <ul className="space-y-1 max-h-64 overflow-auto">
            {allExports.map(f => (
              <li key={f.name} className="flex items-center justify-between text-xs">
                <span className="font-mono text-[#374151]">{f.name.replace(/^exports\//, '')}</span>
                <span className="flex items-center gap-3">
                  <span className="text-[#9ca3af]">{formatBytes(f.size)}</span>
                  <DownloadLink jobId={job.id} file={f} />
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Per-epoch checkpoints */}
      {pthEpochs.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => setExp(v => !v)}
            className="text-xs font-semibold text-[#7E3AF2] hover:underline"
          >
            {expanded ? '▾' : '▸'} Per-epoch checkpoints ({pthEpochs.length})
          </button>
          {expanded && (
            <ul className="mt-2 space-y-1 max-h-64 overflow-auto pl-3">
              {pthEpochs.map(f => (
                <li key={f.name} className="flex items-center justify-between text-xs">
                  <span className="font-mono text-[#374151]">{f.name}</span>
                  <span className="flex items-center gap-3">
                    <span className="text-[#9ca3af]">{formatBytes(f.size)}</span>
                    <DownloadLink jobId={job.id} file={f} />
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Logs */}
      {(trainingLog || otherLogs.length > 0) && (
        <div className="mt-3 pt-3 border-t border-[#f3f4f6] flex items-center gap-3 text-xs">
          {trainingLog && <DownloadLink jobId={job.id} file={trainingLog} label="training.log" />}
          {otherLogs.map(f => (
            <DownloadLink key={f.name} jobId={job.id} file={f} label={f.name} />
          ))}
        </div>
      )}
    </div>
  )
}

function byModifiedDesc(a: FileEntry, b: FileEntry): number {
  const ta = a.modified ? Date.parse(a.modified) : 0
  const tb = b.modified ? Date.parse(b.modified) : 0
  return tb - ta
}

function DownloadCard({
  jobId, file, title, subtitle, accent, extraDownloadFiles,
}: {
  jobId: string
  file: FileEntry | null
  title: string
  subtitle: string
  accent?: boolean
  extraDownloadFiles?: FileEntry[]
}) {
  const disabled = !file
  const href = file
    ? `/api/spark/jobs/${jobId}/files?path=${encodeURIComponent(file.name)}&download=1`
    : undefined

  return (
    <a
      href={href}
      aria-disabled={disabled}
      className={`flex items-center gap-3 p-3 rounded-md border transition-colors ${
        disabled
          ? 'opacity-50 pointer-events-none border-[#e5e7eb] bg-[#fafafa]'
          : accent
            ? 'border-[#e9d5ff] bg-[#faf5ff] hover:bg-[#f3e8ff] cursor-pointer'
            : 'border-[#e5e7eb] hover:bg-[#f9fafb] cursor-pointer'
      }`}
    >
      <div className={`w-9 h-9 rounded flex items-center justify-center flex-shrink-0 ${
        accent ? 'bg-[#7E3AF2] text-white' : 'bg-[#f3e8ff] text-[#7E3AF2]'
      }`}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" x2="12" y1="15" y2="3" />
        </svg>
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold text-[#111827]">{title}</div>
        <div className="text-[11px] text-[#6b7280] truncate">
          {file ? <>{file.name.split('/').pop()}{file.size > 0 && ` · ${formatBytes(file.size)}`}</> : subtitle}
        </div>
      </div>
      {extraDownloadFiles && extraDownloadFiles.length > 0 && (
        <span className="text-[10px] text-[#9ca3af]">+{extraDownloadFiles.length}</span>
      )}
    </a>
  )
}

function DownloadLink({ jobId, file, label }: { jobId: string; file: FileEntry; label?: string }) {
  return (
    <a
      href={`/api/spark/jobs/${jobId}/files?path=${encodeURIComponent(file.name)}&download=1`}
      className="text-[#7E3AF2] hover:underline"
    >
      {label ?? 'Download'}
    </a>
  )
}

function formatBytes(n: number): string {
  if (!n) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}
