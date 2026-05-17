'use client'

/**
 * Live training loss chart. Subscribes to the shared parsed-stream hook
 * (see use-training-stream.ts) — multiple components on the same job page
 * share one EventSource. Renders inline SVG with EMA smoothing, train/val
 * series, best-marker, zoom presets, scroll/drag pan-zoom, and a fullscreen
 * modal.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { useTrainingStream, type Point, type Series } from './use-training-stream'

interface TrainingChartProps {
  jobId: string
}

const ZOOM_PRESETS: { key: string; label: string; epochs: number | null }[] = [
  { key: 'all',     label: 'All',   epochs: null },
  { key: 'last_1',  label: '1 ep',  epochs: 1   },
  { key: 'last_5',  label: '5 ep',  epochs: 5   },
  { key: 'last_10', label: '10 ep', epochs: 10  },
  { key: 'last_20', label: '20 ep', epochs: 20  },
]

const SMOOTH_PRESETS = [0.0, 0.5, 0.9, 0.95, 0.99]

interface CustomView { xLo: number; xHi: number; yMin: number; yMax: number }

export function TrainingChart({ jobId }: TrainingChartProps) {
  const { series, status: streaming } = useTrainingStream(jobId)
  const [smoothing, setSmoothing] = useState(0.6)
  const [zoomKey, setZoomKey]     = useState('all')
  const [showRaw, setShowRaw]     = useState(true)
  const [logScale, setLogScale]   = useState(false)
  // When the user pans/zooms with the mouse, we set customView and stop
  // following the preset. Pressing Reset (or clicking a preset) clears it.
  const [customView, setCustomView] = useState<CustomView | null>(null)
  const [expanded, setExpanded]   = useState(false)

  // ── Smoothing (EMA, matching training_monitor.py:apply_smoothing) ─────────
  const smoothed = useMemo(() => {
    const apply = (pts: Point[]) => {
      if (smoothing <= 0 || pts.length === 0) return pts
      const out: Point[] = []
      let last = pts[0].y
      for (const p of pts) {
        last = smoothing * last + (1 - smoothing) * p.y
        out.push({ x: p.x, y: last })
      }
      return out
    }
    return {
      train:      apply(series.train),
      val:        apply(series.val),
      trainLpips: apply(series.trainLpips),
      valLpips:   apply(series.valLpips),
    }
  }, [series, smoothing])

  // ── Best tracking ────────────────────────────────────────────────────────
  const best = useMemo(() => {
    if (series.train.length === 0) return null
    let bestY = Infinity, bestX = 0
    for (const p of series.train) if (p.y < bestY) { bestY = p.y; bestX = p.x }
    return { x: bestX, y: bestY }
  }, [series.train])

  // ── View extents — preset-derived unless user has dragged/zoomed ─────────
  const view = useMemo<CustomView | null>(() => {
    if (customView) return customView

    const allX = series.train.length > 0 ? series.train : series.val
    if (allX.length === 0) return null
    const xMin = allX[0].x
    const xMax = allX[allX.length - 1].x
    const preset = ZOOM_PRESETS.find(p => p.key === zoomKey)
    let xLo = xMin, xHi = xMax
    if (preset?.epochs != null) xLo = Math.max(xMin, xMax - preset.epochs)

    // Y-bounds: smoothed train + (raw train only when visible) — matches
    // training_monitor.py:1198-1203. The point is to keep the y-axis useful
    // even when the raw curve has a one-time spike that would otherwise
    // dominate. Val series is included similarly (raw counts only when raw
    // overlay is shown, otherwise smoothed val is the floor).
    const yPts: number[] = []
    const collect = (pts: Point[]) => {
      for (const p of pts) if (p.x >= xLo && p.x <= xHi) yPts.push(p.y)
    }
    collect(smoothed.train)
    if (showRaw) collect(series.train)
    if (smoothed.val.length > 0) {
      collect(smoothed.val)
      if (showRaw) collect(series.val)
    }
    if (yPts.length === 0) return null
    let yMin = Math.min(...yPts), yMax = Math.max(...yPts)
    const margin = (yMax - yMin) * 0.1 || 0.001
    yMin = Math.max(0, yMin - margin)
    yMax = yMax + margin
    return { xLo, xHi, yMin, yMax }
  }, [series, smoothed, zoomKey, showRaw, customView])

  // Picking a preset clears the custom view; conversely, mouse interaction
  // creates a custom view that ignores the preset.
  function selectPreset(key: string) {
    setZoomKey(key)
    setCustomView(null)
  }
  function resetView() {
    setCustomView(null)
    setZoomKey('all')
  }

  // Esc closes the expanded modal
  useEffect(() => {
    if (!expanded) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setExpanded(false) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [expanded])

  return (
    <>
      <div className="bg-white border border-[#e5e7eb] rounded-lg overflow-hidden">
        <ChartToolbar
          streaming={streaming}
          series={series}
          best={best}
          zoomKey={customView ? 'custom' : zoomKey}
          setZoomKey={selectPreset}
          logScale={logScale}     setLogScale={setLogScale}
          showRaw={showRaw}       setShowRaw={setShowRaw}
          smoothing={smoothing}   setSmoothing={setSmoothing}
          custom={!!customView}   onReset={resetView}
          onExpand={() => setExpanded(true)}
        />
        <div className="p-3">
          <ChartSvg
            view={view}
            customView={customView}
            setCustomView={setCustomView}
            series={series}
            smoothed={smoothed}
            showRaw={showRaw}
            logScale={logScale}
            best={best}
            streaming={streaming}
            width={760}
            height={260}
          />
        </div>
      </div>

      {/* Fullscreen modal — same chart, much bigger */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
          onClick={() => setExpanded(false)}
        >
          <div
            className="bg-white rounded-lg shadow-2xl flex flex-col"
            style={{ width: '95vw', height: '90vh' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-2 border-b border-[#e5e7eb]">
              <span className="text-sm font-semibold text-[#111827]">Loss Curve — Expanded</span>
              <button
                onClick={() => setExpanded(false)}
                className="text-[#6b7280] hover:text-[#111827] text-xl leading-none"
                title="Close (Esc)"
              >×</button>
            </div>
            <ChartToolbar
              streaming={streaming}
              series={series}
              best={best}
              zoomKey={customView ? 'custom' : zoomKey}
              setZoomKey={selectPreset}
              logScale={logScale}     setLogScale={setLogScale}
              showRaw={showRaw}       setShowRaw={setShowRaw}
              smoothing={smoothing}   setSmoothing={setSmoothing}
              custom={!!customView}   onReset={resetView}
            />
            <div className="flex-1 p-4 overflow-hidden">
              <ChartSvg
                view={view}
                customView={customView}
                setCustomView={setCustomView}
                series={series}
                smoothed={smoothed}
                showRaw={showRaw}
                logScale={logScale}
                best={best}
                streaming={streaming}
                width={1600}
                height={800}
                fillContainer
              />
            </div>
            <div className="px-4 py-2 border-t border-[#e5e7eb] text-[11px] text-[#6b7280] font-mono">
              Scroll to zoom · Drag to pan · Click a preset or Reset to return to follow mode
            </div>
          </div>
        </div>
      )}
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Toolbar
// ─────────────────────────────────────────────────────────────────────────────

interface ToolbarProps {
  streaming:    'connecting' | 'streaming' | 'closed' | 'error'
  series:       Series
  best:         { x: number; y: number } | null
  zoomKey:      string
  setZoomKey:   (k: string) => void
  logScale:     boolean
  setLogScale:  (fn: (v: boolean) => boolean) => void
  showRaw:      boolean
  setShowRaw:   (fn: (v: boolean) => boolean) => void
  smoothing:    number
  setSmoothing: (n: number) => void
  custom:       boolean
  onReset:      () => void
  onExpand?:    () => void
}

function ChartToolbar({
  streaming, series, best, zoomKey, setZoomKey,
  logScale, setLogScale, showRaw, setShowRaw,
  smoothing, setSmoothing, custom, onReset, onExpand,
}: ToolbarProps) {
  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-[#e5e7eb] flex-wrap gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-[#111827]">Loss Curve</span>
        <span className={`text-[11px] font-mono ${
          streaming === 'streaming' ? 'text-[#16A34A]' :
          streaming === 'error' ? 'text-[#EF4444]' : 'text-[#9ca3af]'
        }`}>
          {streaming === 'streaming' ? '● live' :
           streaming === 'closed'    ? '● closed' :
           streaming === 'error'     ? '● error'  : '● connecting…'}
        </span>
        <span className="text-[11px] text-[#6b7280]">
          {series.train.length} pts
          {series.val.length > 0 && ` · ${series.val.length} val`}
          {best && ` · best ${best.y.toFixed(5)} @ ep ${best.x.toFixed(1)}`}
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs flex-wrap justify-end">
        {/* Zoom presets */}
        <div className="flex gap-0.5 bg-[#f3f4f6] rounded p-0.5">
          {ZOOM_PRESETS.map(p => (
            <button
              key={p.key}
              onClick={() => setZoomKey(p.key)}
              className={`px-2 py-0.5 rounded font-medium ${
                zoomKey === p.key && !custom
                  ? 'bg-white text-[#111827] shadow-sm'
                  : 'text-[#6b7280] hover:text-[#374151]'
              }`}
            >
              {p.label}
            </button>
          ))}
          {custom && (
            <button
              onClick={onReset}
              className="px-2 py-0.5 rounded font-medium bg-[#fef3c7] text-[#92400e] hover:bg-[#fde68a]"
              title="Return to preset zoom"
            >
              Reset
            </button>
          )}
        </div>
        <button
          onClick={() => setLogScale(v => !v)}
          className={`px-2 py-1 rounded font-medium border ${
            logScale ? 'bg-[#7E3AF2] text-white border-[#7E3AF2]' : 'border-[#e5e7eb] text-[#6b7280]'
          }`}
        >
          Log Y
        </button>
        <button
          onClick={() => setShowRaw(v => !v)}
          className={`px-2 py-1 rounded font-medium border ${
            showRaw ? 'bg-[#7E3AF2] text-white border-[#7E3AF2]' : 'border-[#e5e7eb] text-[#6b7280]'
          }`}
        >
          Raw
        </button>
        <label className="flex items-center gap-1.5 text-[#6b7280]">
          Smooth
          <input
            type="range" min={0} max={0.99} step={0.01}
            value={smoothing}
            onChange={e => setSmoothing(parseFloat(e.target.value))}
            className="accent-[#7E3AF2]"
            style={{ width: 80 }}
          />
          <span className="font-mono text-[10px] w-7 text-right">{smoothing.toFixed(2)}</span>
          <button
            onClick={() => {
              const next = SMOOTH_PRESETS.find(p => p > smoothing + 0.005) ?? 0
              setSmoothing(next)
            }}
            className="text-[10px] text-[#7E3AF2] hover:underline"
            title="Cycle smoothing presets"
          >
            cycle
          </button>
        </label>
        {onExpand && (
          <button
            onClick={onExpand}
            className="px-2 py-1 rounded font-medium border border-[#e5e7eb] text-[#6b7280] hover:bg-[#f9fafb]"
            title="Expand to fullscreen"
          >
            ⤢ Expand
          </button>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart SVG (also handles its own pointer events for pan/zoom)
// ─────────────────────────────────────────────────────────────────────────────

interface ChartSvgProps {
  view:           CustomView | null
  customView:     CustomView | null
  setCustomView:  (v: CustomView | null) => void
  series:         Series
  smoothed:       Series
  showRaw:        boolean
  logScale:       boolean
  best:           { x: number; y: number } | null
  streaming:      'connecting' | 'streaming' | 'closed' | 'error'
  width:          number
  height:         number
  fillContainer?: boolean
}

const TRAIN_COLOR = '#ae69f4'
const VAL_COLOR   = '#7E3AF2'

function ChartSvg({
  view, customView, setCustomView,
  series, smoothed, showRaw, logScale, best, streaming,
  width: W, height: H, fillContainer,
}: ChartSvgProps) {
  const PAD = { l: 50, r: 16, t: 14, b: 28 }
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const svgRef = useRef<SVGSVGElement | null>(null)

  // hover state lives here so the modal/inline copies don't share it
  const [hover, setHover] = useState<{ x: number; y: number; ep: number; train: number | null; val: number | null } | null>(null)
  // pan state — when set, mouse-move updates customView relative to start
  const panRef = useRef<{ pxStart: number; pyStart: number; viewStart: CustomView } | null>(null)

  function project(p: Point): [number, number] | null {
    if (!view) return null
    const xRange = view.xHi - view.xLo || 1
    const x = PAD.l + ((p.x - view.xLo) / xRange) * plotW
    let yNorm: number
    if (logScale) {
      const lo = Math.max(view.yMin, 1e-6)
      const yLog = Math.log(Math.max(p.y, 1e-6))
      const loLog = Math.log(lo)
      const hiLog = Math.log(view.yMax)
      yNorm = (yLog - loLog) / (hiLog - loLog || 1)
    } else {
      yNorm = (p.y - view.yMin) / (view.yMax - view.yMin || 1)
    }
    const y = PAD.t + (1 - yNorm) * plotH
    return [x, y]
  }

  function path(pts: Point[]): string {
    let d = ''
    for (const p of pts) {
      if (p.x < (view?.xLo ?? -Infinity) || p.x > (view?.xHi ?? Infinity)) continue
      const proj = project(p); if (!proj) continue
      d += d === '' ? `M${proj[0].toFixed(1)},${proj[1].toFixed(1)}` : ` L${proj[0].toFixed(1)},${proj[1].toFixed(1)}`
    }
    return d
  }

  const yTicks = useMemo(() => {
    if (!view) return []
    const ticks: number[] = []
    if (logScale) {
      const lo = Math.max(view.yMin, 1e-6)
      const startExp = Math.floor(Math.log10(lo))
      const endExp   = Math.ceil(Math.log10(view.yMax))
      for (let e = startExp; e <= endExp; e++) ticks.push(Math.pow(10, e))
    } else {
      const n = 4
      for (let i = 0; i <= n; i++) ticks.push(view.yMin + (view.yMax - view.yMin) * (i / n))
    }
    return ticks
  }, [view, logScale])

  // Map a pixel-space point on the SVG back to data space (xLo/Hi/yMin/yMax)
  function pixelToData(px: number, py: number): { x: number; y: number } | null {
    if (!view) return null
    const xRange = view.xHi - view.xLo || 1
    const x = view.xLo + ((px - PAD.l) / plotW) * xRange
    if (logScale) {
      const loLog = Math.log(Math.max(view.yMin, 1e-6))
      const hiLog = Math.log(view.yMax)
      const yLog  = loLog + (1 - (py - PAD.t) / plotH) * (hiLog - loLog)
      return { x, y: Math.exp(yLog) }
    }
    const y = view.yMin + (1 - (py - PAD.t) / plotH) * (view.yMax - view.yMin)
    return { x, y }
  }

  // Convert client (mouse) coords to internal SVG coords (the SVG viewBox is
  // 0..W × 0..H; CSS may render it at a different size).
  function clientToSvg(ev: { clientX: number; clientY: number }): { x: number; y: number } | null {
    const svg = svgRef.current; if (!svg) return null
    const rect = svg.getBoundingClientRect()
    return {
      x: ((ev.clientX - rect.left) / rect.width) * W,
      y: ((ev.clientY - rect.top)  / rect.height) * H,
    }
  }

  // ── Wheel zoom — anchored at cursor, X & Y both ──────────────────────────
  // We attach via useEffect with passive:false so we can preventDefault and
  // not scroll the page while the cursor is over the chart.
  useEffect(() => {
    const svg = svgRef.current; if (!svg) return
    const handler = (ev: WheelEvent) => {
      if (!view) return
      ev.preventDefault()
      const pt = clientToSvg(ev); if (!pt) return
      // ignore wheel outside the plot area
      if (pt.x < PAD.l || pt.x > PAD.l + plotW || pt.y < PAD.t || pt.y > PAD.t + plotH) return
      const data = pixelToData(pt.x, pt.y); if (!data) return

      const factor = ev.deltaY < 0 ? 0.85 : 1.18
      const newWidth  = (view.xHi - view.xLo) * factor
      const newHeight = (view.yMax - view.yMin) * factor

      const relX = (data.x - view.xLo) / (view.xHi - view.xLo || 1)
      const relY = (data.y - view.yMin) / (view.yMax - view.yMin || 1)

      setCustomView({
        xLo: data.x - newWidth * relX,
        xHi: data.x + newWidth * (1 - relX),
        yMin: Math.max(0, data.y - newHeight * relY),
        yMax: data.y + newHeight * (1 - relY),
      })
    }
    svg.addEventListener('wheel', handler, { passive: false })
    return () => svg.removeEventListener('wheel', handler as EventListener)
  // We re-attach on every view change so the closure sees fresh state.
  }, [view, plotW, plotH, PAD.l, PAD.t]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Pan with left-button drag ────────────────────────────────────────────
  function onPointerDown(ev: React.PointerEvent<SVGSVGElement>) {
    if (!view) return
    if (ev.button !== 0) return
    const pt = clientToSvg(ev); if (!pt) return
    if (pt.x < PAD.l || pt.x > PAD.l + plotW || pt.y < PAD.t || pt.y > PAD.t + plotH) return
    panRef.current = { pxStart: pt.x, pyStart: pt.y, viewStart: { ...view } }
    ev.currentTarget.setPointerCapture(ev.pointerId)
  }
  function onPointerUp(ev: React.PointerEvent<SVGSVGElement>) {
    if (panRef.current) {
      panRef.current = null
      try { ev.currentTarget.releasePointerCapture(ev.pointerId) } catch { /* noop */ }
    }
  }

  function onPointerMove(ev: React.PointerEvent<SVGSVGElement>) {
    const pt = clientToSvg(ev); if (!pt) return

    if (panRef.current) {
      const { pxStart, pyStart, viewStart } = panRef.current
      const xRange = viewStart.xHi - viewStart.xLo || 1
      const yRange = viewStart.yMax - viewStart.yMin || 1
      const dxPx = pt.x - pxStart
      const dyPx = pt.y - pyStart
      const dxData = -(dxPx / plotW) * xRange
      const dyData = (dyPx / plotH) * yRange   // SVG y is inverted relative to data y

      // Log Y panning is intentionally disabled (matches training_monitor.py)
      const next: CustomView = {
        xLo:  viewStart.xLo + dxData,
        xHi:  viewStart.xHi + dxData,
        yMin: logScale ? viewStart.yMin : Math.max(0, viewStart.yMin + dyData),
        yMax: logScale ? viewStart.yMax : viewStart.yMax + dyData,
      }
      setCustomView(next)
      return
    }

    // Hover crosshair
    if (!view || series.train.length === 0) return
    if (pt.x < PAD.l || pt.x > PAD.l + plotW) { setHover(null); return }
    const xRange = view.xHi - view.xLo || 1
    const xData  = view.xLo + ((pt.x - PAD.l) / plotW) * xRange
    let nearest = series.train[0], bestD = Infinity
    for (const p of series.train) {
      const d = Math.abs(p.x - xData)
      if (d < bestD) { bestD = d; nearest = p }
    }
    let nearestVal: Point | null = null, bestV = Infinity
    for (const p of series.val) {
      const d = Math.abs(p.x - nearest.x)
      if (d < bestV) { bestV = d; nearestVal = p }
    }
    const proj = project(nearest); if (!proj) return
    setHover({
      x:     proj[0],
      y:     proj[1],
      ep:    nearest.x,
      train: nearest.y,
      val:   nearestVal && Math.abs(nearestVal.x - nearest.x) < 1 ? nearestVal.y : null,
    })
  }

  // Double-click to reset
  function onDoubleClick() { setCustomView(null) }

  const dragging = panRef.current !== null

  return (
    <div className={`relative ${fillContainer ? 'w-full h-full' : ''}`}>
      {!view && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-[#9ca3af]">
          {streaming === 'streaming' ? 'Waiting for first training step…' : 'No training data yet'}
        </div>
      )}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio={fillContainer ? 'none' : 'xMidYMid meet'}
        className={fillContainer ? 'w-full h-full' : 'w-full h-auto'}
        style={{ touchAction: 'none', cursor: dragging ? 'grabbing' : 'crosshair' }}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerMove={onPointerMove}
        onDoubleClick={onDoubleClick}
        onMouseLeave={() => setHover(null)}
      >
        {/* Plot background — also makes the whole plot area receive events */}
        <rect x={PAD.l} y={PAD.t} width={plotW} height={plotH} fill="white" />

        {/* Grid + Y-axis labels */}
        {view && yTicks.map((t, i) => {
          const proj = project({ x: view.xLo, y: t })
          if (!proj) return null
          return (
            <g key={i}>
              <line
                x1={PAD.l} x2={W - PAD.r}
                y1={proj[1]} y2={proj[1]}
                stroke="#e5e7eb" strokeWidth={0.5}
              />
              <text
                x={PAD.l - 6} y={proj[1] + 3}
                textAnchor="end" fontSize={9} fill="#9ca3af"
                fontFamily="monospace"
              >
                {logScale ? t.toExponential(0) : t.toFixed(4)}
              </text>
            </g>
          )
        })}

        {/* X-axis labels (start / end epoch) */}
        {view && (
          <>
            <text x={PAD.l} y={H - 8} fontSize={10} fill="#6b7280">
              ep {view.xLo.toFixed(1)}
            </text>
            <text x={W - PAD.r} y={H - 8} fontSize={10} fill="#6b7280" textAnchor="end">
              ep {view.xHi.toFixed(1)}
            </text>
          </>
        )}

        {/* Plot clip path — keeps lines from drawing into the axis label area */}
        <defs>
          <clipPath id={`plotClip-${W}x${H}`}>
            <rect x={PAD.l} y={PAD.t} width={plotW} height={plotH} />
          </clipPath>
        </defs>

        <g clipPath={`url(#plotClip-${W}x${H})`}>
          {/* Raw train (faint) */}
          {showRaw && view && (
            <path d={path(series.train)} fill="none" stroke={TRAIN_COLOR} strokeWidth={0.6} opacity={0.25} />
          )}
          {/* Smoothed train */}
          {view && (
            <path d={path(smoothed.train)} fill="none" stroke={TRAIN_COLOR} strokeWidth={1.8} />
          )}

          {/* Validation */}
          {showRaw && view && series.val.length > 0 && (
            <path d={path(series.val)} fill="none" stroke={VAL_COLOR} strokeWidth={0.6} opacity={0.3} strokeDasharray="3,2" />
          )}
          {view && smoothed.val.length > 0 && (
            <path d={path(smoothed.val)} fill="none" stroke={VAL_COLOR} strokeWidth={1.6} strokeDasharray="6,4" />
          )}
          {/* Validation points */}
          {view && smoothed.val.map((p, i) => {
            const proj = project(p); if (!proj) return null
            if (p.x < view.xLo || p.x > view.xHi) return null
            return <circle key={i} cx={proj[0]} cy={proj[1]} r={2.5} fill={VAL_COLOR} />
          })}

          {/* Best marker */}
          {view && best && best.x >= view.xLo && best.x <= view.xHi && (() => {
            const proj = project(best); if (!proj) return null
            return (
              <g>
                <circle cx={proj[0]} cy={proj[1]} r={5} fill="#16A34A" stroke="white" strokeWidth={1.5} />
                <text x={proj[0] + 8} y={proj[1] - 6} fontSize={9} fill="#16A34A" fontFamily="monospace">
                  best {best.y.toFixed(5)}
                </text>
              </g>
            )
          })()}
        </g>

        {/* Crosshair + tooltip */}
        {hover && !dragging && (
          <g pointerEvents="none">
            <line x1={hover.x} x2={hover.x} y1={PAD.t} y2={H - PAD.b} stroke="#9ca3af" strokeWidth={0.5} strokeDasharray="2,2" />
            <circle cx={hover.x} cy={hover.y} r={3} fill={TRAIN_COLOR} />
            <g transform={`translate(${Math.min(hover.x + 8, W - 130)}, ${Math.max(hover.y - 30, PAD.t)})`}>
              <rect width={120} height={hover.val !== null ? 36 : 24} fill="white" stroke="#e5e7eb" rx={3} />
              <text x={6} y={14} fontSize={10} fill="#374151" fontFamily="monospace">
                ep {hover.ep.toFixed(2)}  L:{hover.train?.toFixed(5)}
              </text>
              {hover.val !== null && (
                <text x={6} y={28} fontSize={10} fill={VAL_COLOR} fontFamily="monospace">
                  val:{hover.val.toFixed(5)}
                </text>
              )}
            </g>
          </g>
        )}

        {/* Pan/zoom hint when no custom view yet */}
        {view && !customView && (
          <text x={W - PAD.r - 4} y={H - 8} fontSize={9} fill="#cbd5e1" textAnchor="end" fontFamily="monospace">
            scroll = zoom · drag = pan · dbl-click = reset
          </text>
        )}

        {/* Legend — Val only shows once we've actually parsed val lines.
            Matches Python _update_legend which omits runs with no series. */}
        <g transform={`translate(${W - PAD.r - (series.val.length > 0 ? 110 : 50)}, ${PAD.t + 4})`}>
          <line x1={0} x2={14} y1={4} y2={4} stroke={TRAIN_COLOR} strokeWidth={2} />
          <text x={18} y={8} fontSize={9} fill="#6b7280">Train</text>
          {series.val.length > 0 && (
            <>
              <line x1={50} x2={64} y1={4} y2={4} stroke={VAL_COLOR} strokeWidth={2} strokeDasharray="3,2" />
              <text x={68} y={8} fontSize={9} fill="#6b7280">Val</text>
            </>
          )}
        </g>
      </svg>
    </div>
  )
}
