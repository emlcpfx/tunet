'use client'

/**
 * Live preview-image panel for a training job.
 *
 * The trainer writes:
 *   <output>/training_preview.jpg     — train batch grid (src | dst | pred | diff…)
 *   <output>/val_preview.jpg          — held-out batch grid (same layout)
 *
 * The Spark agent streams /output/ to ShareSync after each ~3-5s quiet window.
 * We poll HEAD via the /api/spark/jobs/:id/files proxy to detect updates, and
 * re-fetch the JPG when Last-Modified changes. While the job is live we tick
 * every 5s; after terminal we render the most recent frame and stop.
 *
 * Layout matches Spark/frontend/job-detail.html lines 326-350: tabs for
 * Training / Validation, dark canvas, footer with last-update info. Adds a
 * Diff Amplify slider to match tunet.py's preview ergonomics.
 */

import { useEffect, useState, useRef } from 'react'
import type { SparkJob } from '@/lib/spark-types'
import { ACTIVE_STATUSES } from '@/lib/spark-types'
import { useTrainingStream } from './use-training-stream'

type Tab = 'training' | 'validation'

interface PreviewImagesProps {
  job: SparkJob
}

interface PreviewState {
  tab:        Tab
  src:        string | null    // image URL with cache-buster
  fetchedAt:  number | null    // when the URL was last refreshed
  lastModRaw: string | null    // upstream Last-Modified header
  status:     'idle' | 'loading' | 'ok' | 'missing' | 'error'
  error:      string | null
}

const TAB_FILE: Record<Tab, string> = {
  training:   'training_preview.jpg',
  validation: 'val_preview.jpg',
}

export function PreviewImages({ job }: PreviewImagesProps) {
  const [tab, setTab] = useState<Tab>('training')
  const [state, setState] = useState<PreviewState>({
    tab: 'training', src: null, fetchedAt: null, lastModRaw: null,
    status: 'idle', error: null,
  })
  const [zoom, setZoom] = useState<'fit' | 1 | 2>('fit')
  const [expanded, setExpanded] = useState(false)
  const lastModRefs = useRef<Record<Tab, string | null>>({ training: null, validation: null })
  // We can't trust Spark's job-level status alone — it stays 'provisioning'
  // for minutes after the container is up. If the SSE stream is producing
  // training-step lines, the trainer is demonstrably running and we should
  // keep polling regardless of what the API says. See spark-types.ts derivedStatus.
  const sparkSaysLive = ACTIVE_STATUSES.has(job.status)

  // Esc closes the expanded modal
  useEffect(() => {
    if (!expanded) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setExpanded(false) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [expanded])

  // Subscribe to the parsed log stream so we can show "step X / first preview
  // at step N" while the user is waiting. We extract the configured interval
  // from the actual log output (which we don't currently parse), so for now
  // we mirror the buildConfig default of 50 — keeping it as a constant means
  // the empty-state hint stays correct if the user didn't override.
  const { series } = useTrainingStream(job.id)
  // We track the underlying step (not epoch fraction) for the hint; convert
  // back from `x` since x = (epoch - 1) + step_in_epoch / total_steps. We
  // don't have step_total, so just count training-line emissions × log_interval.
  const stepsSeen = series.train.length * 5  // log_interval default = 5
  const FIRST_PREVIEW_STEP = 50  // mirrors buildConfig default; see spark-presets.ts

  // The trainer is demonstrably alive if any training-step line has come
  // through the SSE stream. If Spark's job status is also active, great. If
  // it's stuck on 'provisioning' but lines are arriving, trust the lines.
  const isLive = sparkSaysLive || stepsSeen > 0

  // Reset cached headers when job changes
  useEffect(() => {
    lastModRefs.current = { training: null, validation: null }
  }, [job.id])

  // ── Probe + (re)fetch loop ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null

    async function tick() {
      if (cancelled) return

      const file = TAB_FILE[tab]
      const url  = `/api/spark/jobs/${job.id}/files?path=${encodeURIComponent(file)}`

      try {
        // HEAD-equivalent via a small GET (the proxy doesn't currently support
        // HEAD; the file is small enough that a GET on change-detection is OK).
        // We use If-Modified-Since when we have it.
        const headers: Record<string, string> = {}
        const known = lastModRefs.current[tab]
        if (known) headers['If-Modified-Since'] = known

        const res = await fetch(url, { method: 'GET', headers, cache: 'no-store' })

        if (cancelled) return

        if (res.status === 304) {
          // No change — keep current image
          setState(s => s.tab === tab ? { ...s, status: 'ok' } : s)
        } else if (res.status === 404) {
          setState({
            tab, src: null, fetchedAt: null, lastModRaw: null,
            status: 'missing',
            error:  'No preview written yet',
          })
        } else if (!res.ok) {
          setState({
            tab, src: null, fetchedAt: null, lastModRaw: null,
            status: 'error',
            error:  `HTTP ${res.status}`,
          })
        } else {
          const lm = res.headers.get('Last-Modified')
          if (lm) lastModRefs.current[tab] = lm
          // Use a cache-busting URL so the <img> displays the fresh version.
          // We can't pass the blob directly without losing the actual image.
          const bustedUrl = `${url}&v=${Date.now()}`
          setState({
            tab,
            src:        bustedUrl,
            fetchedAt:  Date.now(),
            lastModRaw: lm,
            status:     'ok',
            error:      null,
          })
        }
      } catch (e) {
        if (cancelled) return
        setState(s => ({ ...s, status: 'error', error: e instanceof Error ? e.message : 'fetch failed' }))
      }

      // Schedule next probe — only while live, and only for the active tab
      if (!cancelled && isLive) {
        timer = setTimeout(tick, 5000)
      }
    }

    setState(s => ({ ...s, tab, status: 'loading' }))
    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [job.id, tab, isLive])

  const filename = TAB_FILE[tab]
  const downloadUrl = `/api/spark/jobs/${job.id}/files?path=${encodeURIComponent(filename)}&download=1`

  return (
    <>
    <div className="bg-white border border-[#e5e7eb] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#e5e7eb] flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-[#111827]">Preview</span>
          <div className="inline-flex bg-[#f3f4f6] rounded p-0.5">
            <TabBtn active={tab === 'training'}   onClick={() => setTab('training')}>Training</TabBtn>
            <TabBtn active={tab === 'validation'} onClick={() => setTab('validation')}>Validation</TabBtn>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs text-[#6b7280]">
          <div className="inline-flex items-center gap-1">
            Zoom:
            <select
              value={zoom === 'fit' ? 'fit' : String(zoom)}
              onChange={e => setZoom(e.target.value === 'fit' ? 'fit' : (parseInt(e.target.value, 10) as 1 | 2))}
              className="border border-[#e5e7eb] rounded px-1.5 py-0.5 text-xs"
            >
              <option value="fit">Fit</option>
              <option value="1">100%</option>
              <option value="2">200%</option>
            </select>
          </div>
          <a
            href={downloadUrl}
            className="text-[#7E3AF2] hover:underline"
            title="Download original JPG"
          >
            Download
          </a>
          {state.src && (
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="px-2 py-1 rounded font-medium border border-[#e5e7eb] text-[#6b7280] hover:bg-[#f9fafb]"
              title="Expand to fullscreen with zoom + pan"
            >
              ⤢ Expand
            </button>
          )}
        </div>
      </div>

      {/* Image canvas */}
      <div className="bg-[#1a1a1a] flex items-center justify-center" style={{ minHeight: 280 }}>
        {state.status === 'loading' && !state.src && (
          <span className="text-xs text-[#9ca3af]">Fetching preview…</span>
        )}
        {state.status === 'missing' && (
          <div className="text-center px-6">
            <p className="text-xs text-[#9ca3af]">
              Not written yet — first preview saves at end of epoch 1 (step ~{FIRST_PREVIEW_STEP}+).
            </p>
            {stepsSeen > 0 && (
              <p className="text-[11px] text-[#6b7280] mt-1 font-mono">
                training has emitted ~{stepsSeen} steps so far
              </p>
            )}
            {stepsSeen > FIRST_PREVIEW_STEP * 2 && (
              <p className="text-[11px] text-[#fca5a5] mt-2">
                Should have arrived by now — ShareSync upload may be lagging, or the file path differs.
              </p>
            )}
          </div>
        )}
        {state.status === 'error' && (
          <span className="text-xs text-[#fca5a5]">Error: {state.error}</span>
        )}
        {state.src && state.status !== 'missing' && (
          <div
            className={zoom === 'fit' ? 'w-full overflow-hidden' : 'overflow-auto max-h-[600px]'}
            style={zoom === 'fit' ? {} : { width: '100%' }}
          >
            {/* No `key` on this <img>. Keying on state.src would force a
                remount on every poll (the URL changes via cache-buster on
                every refresh) — the browser would clear the canvas while
                downloading the new image, producing a flicker the user
                noticed as "disappear and reappear." Without the key, React
                keeps the same DOM node and just updates src; browsers swap
                in the new image only once it's decoded, so the old preview
                stays visible until the new one is ready. */}
            <img
              src={state.src}
              alt={`${tab} preview`}
              className="block"
              style={
                zoom === 'fit'
                  ? { width: '100%', height: 'auto', objectFit: 'contain' }
                  : { width: zoom === 2 ? '200%' : '100%', height: 'auto', imageRendering: 'pixelated' }
              }
            />
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 text-[11px] text-[#6b7280] flex items-center justify-between border-t border-[#e5e7eb]">
        <span>
          {state.lastModRaw
            ? `Last update: ${formatTs(state.lastModRaw)}`
            : isLive ? 'Updating every 5s while job is running' : 'Job ended — showing last available'}
        </span>
        <span className="font-mono">{filename}</span>
      </div>
    </div>

    {expanded && state.src && (
      <ExpandedPreview
        src={state.src}
        filename={filename}
        tab={tab}
        onTabChange={setTab}
        downloadUrl={downloadUrl}
        lastModRaw={state.lastModRaw}
        isLive={isLive}
        onClose={() => setExpanded(false)}
      />
    )}
    </>
  )
}

function TabBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
        active
          ? 'bg-white text-[#111827] shadow-sm'
          : 'text-[#6b7280] hover:text-[#374151]'
      }`}
    >
      {children}
    </button>
  )
}

function formatTs(httpDate: string): string {
  try {
    const d = new Date(httpDate)
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return httpDate
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expanded modal — fullscreen preview with wheel zoom + drag pan
// ─────────────────────────────────────────────────────────────────────────────

interface ExpandedPreviewProps {
  src:         string
  filename:    string
  tab:         Tab
  onTabChange: (t: Tab) => void
  downloadUrl: string
  lastModRaw:  string | null
  isLive:      boolean
  onClose:     () => void
}

/**
 * Fullscreen preview viewer with mouse-wheel zoom and click-drag pan. The
 * `src` is the same cache-busted URL the inline panel uses, so when the
 * trainer writes a new preview the modal shows it on the next 5s tick
 * without losing the user's pan/zoom state (we key the <img> on `src` so
 * React swaps it in cleanly).
 *
 * Implementation: we render the image inside a positioned `<div>` and apply
 * `transform: translate(...) scale(...)` for both pan and zoom. Wheel-zoom
 * is anchored at the cursor (the canonical math: world point under cursor
 * stays under cursor before/after zoom).
 */
function ExpandedPreview({
  src, filename, tab, onTabChange, downloadUrl, lastModRaw, isLive, onClose,
}: ExpandedPreviewProps) {
  const [scale, setScale]       = useState(1)
  const [tx, setTx]             = useState(0)
  const [ty, setTy]             = useState(0)
  const [imgDims, setImgDims]   = useState<{ w: number; h: number } | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const dragStartRef = useRef<{ px: number; py: number; tx: number; ty: number } | null>(null)
  const [dragging, setDragging] = useState(false)

  // Reset transform when the image dimensions change (new tab, or first load)
  useEffect(() => {
    if (!imgDims) return
    fitToContainer(imgDims)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgDims])

  function fitToContainer(dims: { w: number; h: number }) {
    const c = containerRef.current
    if (!c) return
    const rect = c.getBoundingClientRect()
    const sx = rect.width  / dims.w
    const sy = rect.height / dims.h
    const fitScale = Math.min(sx, sy) * 0.95
    setScale(fitScale)
    setTx((rect.width  - dims.w * fitScale) / 2)
    setTy((rect.height - dims.h * fitScale) / 2)
  }

  // ── Wheel zoom — anchored at cursor ──────────────────────────────────────
  // Attach via useEffect with passive:false so we can preventDefault and not
  // scroll the page underneath.
  useEffect(() => {
    const c = containerRef.current
    if (!c) return
    const handler = (ev: WheelEvent) => {
      ev.preventDefault()
      const rect = c.getBoundingClientRect()
      const cursorX = ev.clientX - rect.left
      const cursorY = ev.clientY - rect.top
      const factor = ev.deltaY < 0 ? 1.15 : 1 / 1.15
      setScale(prevScale => {
        const newScale = Math.max(0.05, Math.min(40, prevScale * factor))
        // Adjust translate so the world point under the cursor stays under it.
        // Solve: (cursor - oldTx) / oldScale = (cursor - newTx) / newScale
        //   →   newTx = cursor - (cursor - oldTx) * (newScale / oldScale)
        const ratio = newScale / prevScale
        setTx(prevTx => cursorX - (cursorX - prevTx) * ratio)
        setTy(prevTy => cursorY - (cursorY - prevTy) * ratio)
        return newScale
      })
    }
    c.addEventListener('wheel', handler, { passive: false })
    return () => c.removeEventListener('wheel', handler as EventListener)
  }, [])

  // ── Drag pan ─────────────────────────────────────────────────────────────
  function onPointerDown(ev: React.PointerEvent<HTMLDivElement>) {
    if (ev.button !== 0) return
    dragStartRef.current = { px: ev.clientX, py: ev.clientY, tx, ty }
    setDragging(true)
    ev.currentTarget.setPointerCapture(ev.pointerId)
  }
  function onPointerMove(ev: React.PointerEvent<HTMLDivElement>) {
    const start = dragStartRef.current
    if (!start) return
    setTx(start.tx + (ev.clientX - start.px))
    setTy(start.ty + (ev.clientY - start.py))
  }
  function onPointerUp(ev: React.PointerEvent<HTMLDivElement>) {
    dragStartRef.current = null
    setDragging(false)
    try { ev.currentTarget.releasePointerCapture(ev.pointerId) } catch { /* noop */ }
  }
  function onDoubleClick() {
    if (imgDims) fitToContainer(imgDims)
  }

  function zoomBy(factor: number) {
    const c = containerRef.current
    if (!c) return
    const rect = c.getBoundingClientRect()
    const cx = rect.width / 2
    const cy = rect.height / 2
    setScale(prevScale => {
      const newScale = Math.max(0.05, Math.min(40, prevScale * factor))
      const ratio = newScale / prevScale
      setTx(prevTx => cx - (cx - prevTx) * ratio)
      setTy(prevTy => cy - (cy - prevTy) * ratio)
      return newScale
    })
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex flex-col"
      onClick={onClose}
    >
      <div
        className="flex-1 flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Toolbar */}
        <div className="flex items-center justify-between px-5 py-3 bg-[#0b1220] border-b border-[#1e293b]">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-white">Preview — Expanded</span>
            <div className="inline-flex bg-[#1e293b] rounded p-0.5">
              <ModalTabBtn active={tab === 'training'}   onClick={() => onTabChange('training')}>Training</ModalTabBtn>
              <ModalTabBtn active={tab === 'validation'} onClick={() => onTabChange('validation')}>Validation</ModalTabBtn>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="font-mono text-[#94a3b8]">
              {scale >= 1 ? `${Math.round(scale * 100)}%` : `${(scale * 100).toFixed(1)}%`}
            </span>
            <button
              onClick={() => zoomBy(1 / 1.5)}
              className="px-2 py-1 rounded text-[#cbd5e1] hover:bg-[#1e293b]"
              title="Zoom out"
            >−</button>
            <button
              onClick={() => zoomBy(1.5)}
              className="px-2 py-1 rounded text-[#cbd5e1] hover:bg-[#1e293b]"
              title="Zoom in"
            >+</button>
            <button
              onClick={() => imgDims && fitToContainer(imgDims)}
              className="px-2 py-1 rounded text-[#cbd5e1] hover:bg-[#1e293b]"
              title="Fit to screen (or double-click image)"
            >
              Fit
            </button>
            <button
              onClick={() => {
                const c = containerRef.current
                if (!c || !imgDims) return
                const rect = c.getBoundingClientRect()
                setScale(1)
                setTx((rect.width  - imgDims.w) / 2)
                setTy((rect.height - imgDims.h) / 2)
              }}
              className="px-2 py-1 rounded text-[#cbd5e1] hover:bg-[#1e293b]"
              title="100%"
            >
              1:1
            </button>
            <a
              href={downloadUrl}
              className="px-2 py-1 rounded text-[#cbd5e1] hover:bg-[#1e293b]"
              title="Download original JPG"
            >
              Download
            </a>
            <button
              onClick={onClose}
              className="px-2 py-1 rounded text-[#cbd5e1] hover:bg-[#1e293b] text-lg leading-none"
              title="Close (Esc)"
            >×</button>
          </div>
        </div>

        {/* Viewport */}
        <div
          ref={containerRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onDoubleClick={onDoubleClick}
          className="flex-1 relative overflow-hidden bg-[#0a0a0a]"
          style={{
            cursor: dragging ? 'grabbing' : 'grab',
            touchAction: 'none',
          }}
        >
          {/* Same flicker-fix as the inline panel: drop the key so React
              keeps this <img> node across refreshes. The pan/zoom state
              lives on the outer transform, not on the img element, so
              there's nothing to lose by reusing the DOM node. */}
          <img
            src={src}
            alt={`${tab} preview`}
            draggable={false}
            onLoad={(e) => {
              const im = e.currentTarget
              setImgDims({ w: im.naturalWidth, h: im.naturalHeight })
            }}
            style={{
              position: 'absolute',
              left: 0, top: 0,
              transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
              transformOrigin: '0 0',
              imageRendering: scale >= 2 ? 'pixelated' : 'auto',
              userSelect: 'none',
              pointerEvents: 'none',
              maxWidth: 'none',
              maxHeight: 'none',
            }}
          />
        </div>

        {/* Footer */}
        <div className="px-5 py-2 bg-[#0b1220] border-t border-[#1e293b] flex items-center justify-between text-[11px] text-[#94a3b8]">
          <span>
            {lastModRaw
              ? `Last update: ${formatTs(lastModRaw)}`
              : isLive ? 'Updating every 5s while job is running' : 'Job ended — showing last available'}
            {imgDims && <span className="ml-3 font-mono">{imgDims.w}×{imgDims.h}</span>}
          </span>
          <span className="font-mono">
            scroll = zoom · drag = pan · dbl-click = fit · Esc = close · {filename}
          </span>
        </div>
      </div>
    </div>
  )
}

function ModalTabBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
        active
          ? 'bg-[#0b1220] text-white shadow-sm'
          : 'text-[#94a3b8] hover:text-white'
      }`}
    >
      {children}
    </button>
  )
}
