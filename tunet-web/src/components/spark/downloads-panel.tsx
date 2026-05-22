'use client'

/**
 * Downloads panel — lists output files (checkpoints + exports) from ShareSync
 * and offers downloads. Mirrors the local Export tab (gui/export_tab.py): the
 * "Export for Flame / AE" and "Export for Nuke" cards are *actions*, not passive
 * links — pick a checkpoint, click, and a small CPU Spark job converts the .pth
 * (Flame ONNX / Nuke TorchScript) and the file auto-downloads when it lands.
 * Recent auto-exports are also surfaced as direct downloads. Per-epoch .pth and
 * the full exports/ tree live in collapsible sub-lists.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import type { SparkJob } from '@/lib/spark-types'

interface FileEntry {
  name:     string
  size:     number
  modified: string | null
}

interface DownloadsPanelProps {
  job: SparkJob
}

type ExportFormat = 'flame' | 'nuke'

// Per-format export lifecycle. Each card tracks its own — the convert job emits
// both deliverables, but the user clicks them independently.
type ExportPhase =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'requested' }                  // running job is exporting inline on its training node
  | { kind: 'converting'; jobId: string }  // spawned CPU job is converting
  | { kind: 'ready' }                      // export is in ShareSync — show a Download link
  | { kind: 'timeout';    jobId?: string }
  | { kind: 'error';      msg: string }

const POLL_MS          = 5_000
const POLL_DEADLINE    = 7 * 60_000    // spawned CPU job: ~2-3 min, give headroom
const REQUEST_DEADLINE = 20 * 60_000   // inline export waits for the next epoch boundary

export function DownloadsPanel({ job }: DownloadsPanelProps) {
  // The app's job pages live at /jobs/[id]; export-job progress links there.
  const jobsBase  = '/jobs'

  // A live job can export on its own training node (no new machine); a finished
  // one can't (model unloaded), so it falls back to a spawned convert job.
  const jobLive = !['succeeded', 'completed', 'failed', 'cancelled', 'stopped', 'error']
    .includes(String(job.status ?? '').toLowerCase())

  const [files, setFiles]     = useState<FileEntry[] | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [loading, setLoad]    = useState(true)
  const [expanded, setExp]    = useState(false)
  const [selectedCkpt, setSelectedCkpt] = useState('')
  const [phases, setPhases]   = useState<Record<ExportFormat, ExportPhase>>({
    flame: { kind: 'idle' }, nuke: { kind: 'idle' },
  })
  const cancelledRef = useRef(false)

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

  // Stop any in-flight export polling if the panel unmounts.
  useEffect(() => () => { cancelledRef.current = true }, [])

  // ── Classify (memoized so the picker effect can depend on pthLatest) ────────
  // Exports land under exports/<flame|nuke>/… (the files route descends one
  // level), so classify by that subfolder and surface the most-recent.
  const flameOnnx = useMemo(
    () => (files ?? []).filter(f => /(^|\/)flame\//i.test(f.name) && f.name.endsWith('.onnx')).sort(byModifiedDesc)[0] ?? null,
    [files],
  )
  // Nuke's deliverable is a trio (.cat Cattery model + .nk node + .pt
  // TorchScript). Group them, surface the .cat as primary.
  const nukeFiles = useMemo(
    () => (files ?? []).filter(f => /(^|\/)nuke\//i.test(f.name) && /\.(cat|nk|pt)$/i.test(f.name)).sort(byModifiedDesc),
    [files],
  )
  const nukePrimary = useMemo(() => nukeFiles.find(f => f.name.endsWith('.cat')) ?? nukeFiles[0] ?? null, [nukeFiles])
  const allExports  = useMemo(() => (files ?? []).filter(f => f.name.startsWith('exports/')).sort(byModifiedDesc), [files])
  const pthAll      = useMemo(() => (files ?? []).filter(f => f.name.endsWith('.pth')).sort(byModifiedDesc), [files])
  const pthLatest   = useMemo(
    () => (files ?? []).find(f => /_latest\.pth$/.test(f.name)) ?? pthAll[0] ?? null,
    [files, pthAll],
  )
  const pthEpochs   = useMemo(() => pthAll.filter(f => f.name !== pthLatest?.name), [pthAll, pthLatest])
  const previewJpgs = useMemo(() => (files ?? []).filter(f => /preview/i.test(f.name) && /\.jpe?g$/i.test(f.name)), [files])
  const trainingLog = useMemo(() => (files ?? []).find(f => f.name === 'training.log') ?? null, [files])
  const otherLogs   = useMemo(() => (files ?? []).filter(f => f.name.endsWith('.log') && f.name !== 'training.log'), [files])

  // Default the picker to the latest checkpoint once files load.
  useEffect(() => {
    if (!selectedCkpt && pthLatest) setSelectedCkpt(pthLatest.name)
  }, [pthLatest, selectedCkpt])

  const canExport = !!pthLatest

  function setPhase(fmt: ExportFormat, p: ExportPhase) {
    setPhases(prev => ({ ...prev, [fmt]: p }))
  }

  // Programmatic download through the files proxy (the raw ShareSync URL needs a
  // bearer the browser doesn't have; the proxy adds it server-side).
  function triggerDownload(relPath: string) {
    const a = document.createElement('a')
    a.href = `/api/spark/jobs/${job.id}/files?path=${encodeURIComponent(relPath)}&download=1`
    a.download = relPath.split('/').pop() ?? 'download'
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  function downloadMany(paths: string[]) {
    // Stagger so the browser doesn't collapse rapid-fire downloads into one.
    paths.forEach((p, i) => setTimeout(() => triggerDownload(p), i * 400))
  }

  // Every file in a format's most-recent export set. Flame ships .onnx + .json
  // and Nuke ships .pt + .nk (+ .cat once built in Nuke) — you need the whole
  // set, not just one file. Grouped by shared base name (epoch + timestamp).
  // Takes an explicit list so callers can use a freshly-fetched listing rather
  // than the (possibly stale) `files` state.
  function exportSetFrom(list: FileEntry[], fmt: ExportFormat): string[] {
    const sub = `exports/${fmt}/`
    const inSub = list.filter(f => f.name.startsWith(sub)).sort(byModifiedDesc)
    if (inSub.length === 0) return []
    const baseOf = (n: string) => n.replace(/\.[^/.]+$/, '')   // strip extension
    const newest = baseOf(inSub[0].name)
    return inSub.filter(f => baseOf(f.name) === newest).map(f => f.name)
  }

  function downloadSet(fmt: ExportFormat) {
    const set = exportSetFrom(files ?? [], fmt)
    if (set.length) downloadMany(set)
  }

  // Click handler for an export card. Preferred path for a RUNNING job: ask its
  // training node to export the current model inline (it already holds the
  // weights — no new machine), then poll for the file and surface a download.
  // Finished job: fall back to a spawned CPU job that loads the picked .pth.
  async function runExport(fmt: ExportFormat) {
    const subdir = fmt === 'nuke' ? 'nuke' : 'flame'
    // Snapshot existing files for this format so we can spot the one the export
    // writes (filenames embed epoch + timestamp, so a fresh export is new).
    const before = new Set((files ?? []).filter(f => f.name.startsWith(`exports/${subdir}/`)).map(f => f.name))
    setPhase(fmt, { kind: 'submitting' })

    if (jobLive) {
      try {
        const res = await fetch(`/api/spark/jobs/${job.id}/request-export`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ flame: fmt === 'flame', nuke: fmt === 'nuke' }),
        })
        const data = await res.json()
        if (res.ok) {
          // The training loop picks up the request at its next epoch boundary
          // and self-uploads the export. Poll the listing for it.
          setPhase(fmt, { kind: 'requested' })
          void pollForExport(fmt, null, subdir, before)
          return
        }
        // notLive (409) means the job finished between render and click — drop
        // to the spawn path. Anything else is a real error.
        if (!(res.status === 409 && data.notLive)) {
          throw new Error(data.error ?? `HTTP ${res.status}`)
        }
      } catch (e) {
        setPhase(fmt, { kind: 'error', msg: e instanceof Error ? e.message : 'export request failed' })
        return
      }
    }

    // Spawn path: load the picked checkpoint on a fresh CPU job and convert it.
    const ckpt = selectedCkpt || pthLatest?.name
    if (!ckpt) { setPhase(fmt, { kind: 'error', msg: 'no checkpoint to export' }); return }
    try {
      const res = await fetch(`/api/spark/jobs/${job.id}/export-onnx`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ checkpointName: ckpt, format: fmt, force: true }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
      if (data.fromCache && data.downloadPath) {
        const entries = await refreshFiles()          // fresh listing → full set
        downloadMany(exportSetFrom(entries, fmt))     // best-effort (within the click's gesture window)
        setPhase(fmt, { kind: 'ready' })
        return
      }
      setPhase(fmt, { kind: 'converting', jobId: data.jobId })
      void pollForExport(fmt, data.jobId, subdir, before)
    } catch (e) {
      setPhase(fmt, { kind: 'error', msg: e instanceof Error ? e.message : 'export failed' })
    }
  }

  async function refreshFiles(): Promise<FileEntry[]> {
    try {
      const res = await fetch(`/api/spark/jobs/${job.id}/files`, { cache: 'no-store' })
      const data = await res.json()
      if (res.ok) {
        const entries = (data.entries ?? []) as FileEntry[]
        setFiles(entries)
        return entries
      }
    } catch { /* keep prior listing */ }
    return files ?? []
  }

  // Poll the listing until the convert job's deliverable appears, then download
  // it. The job emits both formats; we wait for the one this card asked for.
  async function pollForExport(fmt: ExportFormat, jobId: string | null, subdir: string, before: Set<string>) {
    const deadline = Date.now() + (jobId ? POLL_DEADLINE : REQUEST_DEADLINE)
    const tick = async () => {
      if (cancelledRef.current) return
      const entries = await refreshFiles()
      const fresh = entries.filter(f => f.name.startsWith(`exports/${subdir}/`) && !before.has(f.name))
      // Wait for the deliverable of this format (.onnx for flame; .pt for nuke).
      const arrived = fmt === 'nuke'
        ? fresh.some(f => f.name.endsWith('.pt'))
        : fresh.some(f => f.name.endsWith('.onnx'))
      if (arrived) {
        // Grab the whole fresh set (.onnx + .json, or .pt + .nk), not just one.
        // Best-effort — browsers block downloads fired long after a click (no
        // user gesture), so the green Download button (a real gesture) is the
        // reliable path.
        downloadMany(fresh.map(f => f.name))
        setPhase(fmt, { kind: 'ready' })
        return
      }
      if (Date.now() > deadline) { setPhase(fmt, { kind: 'timeout', jobId: jobId ?? undefined }); return }
      setTimeout(() => { void tick() }, POLL_MS)
    }
    setTimeout(() => { void tick() }, POLL_MS)
  }

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

  const ckptLabel = (selectedCkpt || pthLatest?.name || '').split('/').pop() ?? ''

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg p-4">
      <h3 className="text-sm font-semibold text-[#111827] mb-1">Downloads</h3>
      <p className="text-xs text-[#6b7280] mb-3">
        Latest checkpoint and on-demand exports. Pick a checkpoint, then{' '}
        <span className="font-semibold">Export for Flame / AE</span> (ONNX) or{' '}
        <span className="font-semibold">Nuke</span> (TorchScript) — a CPU job converts it
        (~2-3 min, ~$0.02); when it&apos;s ready the card shows a <span className="font-semibold">Download</span> link.
      </p>

      {/* ── Export source ──────────────────────────────────────────────────── */}
      {/* Running job: export runs on its training node (current model, no new
          machine), so the picker doesn't apply. Finished job: pick which saved
          .pth a spawned convert job should load. */}
      {canExport && jobLive && (
        <div className="flex items-center gap-2 mb-3 p-2 rounded-md bg-[#faf5ff] border border-[#e9d5ff] text-xs text-[#6b7280]">
          <span className="text-[#7E3AF2] font-semibold">●</span>
          Running job — export runs on the training node (current model), no new machine.
        </div>
      )}
      {canExport && !jobLive && (
        <div className="flex items-center gap-2 mb-3 p-2 rounded-md bg-[#faf5ff] border border-[#e9d5ff]">
          <label className="text-xs font-semibold text-[#374151] whitespace-nowrap">Checkpoint to export</label>
          <select
            value={selectedCkpt}
            onChange={e => setSelectedCkpt(e.target.value)}
            className="flex-1 min-w-0 text-xs font-mono border border-[#e9d5ff] rounded px-2 py-1.5 bg-white focus:border-[#7E3AF2] focus:outline-none"
          >
            {pthLatest && <option value={pthLatest.name}>{pthLatest.name} (latest)</option>}
            {pthEpochs.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
          </select>
        </div>
      )}

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
          file={previewJpgs[0] ?? null}
          title="Preview images"
          subtitle={previewJpgs.length > 0 ? `${previewJpgs.length} files (download all from list below)` : 'Not generated yet'}
          extraDownloadFiles={previewJpgs.length > 1 ? previewJpgs.slice(1) : undefined}
        />
        <ExportCard
          title="Export for Flame / After Effects"
          hint="ONNX (.onnx + .json)"
          phase={phases.flame}
          existing={flameOnnx}
          canExport={canExport}
          jobLive={jobLive}
          ckptLabel={ckptLabel}
          jobsBase={jobsBase}
          onExport={() => void runExport('flame')}
          onDownload={() => downloadSet('flame')}
        />
        <ExportCard
          title="Export for Nuke"
          hint="TorchScript (.pt + .nk)"
          phase={phases.nuke}
          existing={nukePrimary}
          extraCount={Math.max(0, nukeFiles.length - 1)}
          canExport={canExport}
          jobLive={jobLive}
          ckptLabel={ckptLabel}
          jobsBase={jobsBase}
          onExport={() => void runExport('nuke')}
          onDownload={() => downloadSet('nuke')}
        />
      </div>

      {/* Every file in the exports/ tree — each individually downloadable. */}
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
        <DownloadIcon />
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

/**
 * Action card for an on-demand export. The whole card is the convert button
 * (mirrors the local Export tab); a secondary "download latest" link appears
 * when an export already exists so you can grab it without re-converting.
 */
function ExportCard({
  title, hint, phase, existing, extraCount, canExport, jobLive, ckptLabel, jobsBase, onExport, onDownload,
}: {
  title: string
  hint: string
  phase: ExportPhase
  existing: FileEntry | null
  extraCount?: number
  canExport: boolean
  jobLive: boolean
  ckptLabel: string
  jobsBase: string
  onExport: () => void
  onDownload: () => void   // downloads the format's full export set
}) {
  const busy  = phase.kind === 'submitting' || phase.kind === 'requested' || phase.kind === 'converting'
  const ready = phase.kind === 'ready'
  // Only the spawned-job paths have a job to link to (inline export has none).
  const progressJobId =
    phase.kind === 'converting' ? phase.jobId
    : phase.kind === 'timeout'  ? phase.jobId
    : undefined

  return (
    <div className={`flex items-center gap-3 p-3 rounded-md border transition-colors ${
      busy ? 'border-[#e9d5ff] bg-[#faf5ff]' : ready ? 'border-[#bbf7d0] bg-[#f0fdf4]' : 'border-[#e5e7eb]'
    } ${!canExport ? 'opacity-50' : ''}`}>
      <button
        type="button"
        onClick={onExport}
        disabled={!canExport || busy}
        className="flex items-center gap-3 min-w-0 flex-1 text-left disabled:cursor-not-allowed"
      >
        <div className={`w-9 h-9 rounded flex items-center justify-center flex-shrink-0 ${
          ready ? 'bg-[#16a34a] text-white' : 'bg-[#f3e8ff] text-[#7E3AF2]'
        }`}>
          {busy
            ? <span className="w-4 h-4 border-2 border-[#7E3AF2] border-t-transparent rounded-full animate-spin" />
            : ready
              ? <CheckIcon />
              : <DownloadIcon />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-[#111827]">{title}</div>
          <div className="text-[11px] text-[#6b7280] truncate">
            {phase.kind === 'idle' && (canExport
              ? (jobLive
                  ? <>Export current model on the training node → {hint}</>
                  : <>Convert <span className="font-mono">{ckptLabel}</span> → {hint}</>)
              : 'Waiting for the first checkpoint…')}
            {phase.kind === 'submitting' && 'Requesting export…'}
            {phase.kind === 'requested'  && 'Exporting on the training node — ready at the next epoch (usually ~1-2 min)…'}
            {phase.kind === 'converting' && 'Converting on Spark… (~2-3 min)'}
            {phase.kind === 'ready'    && <span className="text-[#16a34a]">Ready — download it →  ·  click to re-export</span>}
            {phase.kind === 'timeout' && 'Still working — appears here when it completes'}
            {phase.kind === 'error'   && <span className="text-[#EF4444]">{phase.msg}</span>}
            {phase.kind === 'idle' && existing && <span className="text-[#9ca3af]"> · last: {existing.name.split('/').pop()}{extraCount ? ` (+${extraCount})` : ''}</span>}
          </div>
        </div>
      </button>
      <div className="flex flex-col items-end gap-1 flex-shrink-0">
        {progressJobId && (
          <Link href={`${jobsBase}/${progressJobId}`} className="text-[10px] text-[#7E3AF2] hover:underline">
            progress →
          </Link>
        )}
        {ready && (
          <button
            type="button"
            onClick={onDownload}
            className="text-xs font-semibold px-2.5 py-1 rounded-md bg-[#16a34a] text-white hover:bg-[#15803d]"
          >
            Download ↓
          </button>
        )}
        {existing && !busy && !ready && (
          <button type="button" onClick={onDownload} className="text-[10px] text-[#7E3AF2] hover:underline">
            ↓ latest
          </button>
        )}
      </div>
    </div>
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

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function formatBytes(n: number): string {
  if (!n) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}
