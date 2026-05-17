'use client'

/**
 * Preview Filter dialog — equivalent of tunet.py's _show_skip_filter_preview.
 *
 * Runs the patch-skip scan entirely in the browser via a Web Worker. The
 * folder picker already gave us File handles for src/ and dst/ — no upload
 * needed. Patches stream back as they're scored, so the user sees the grid
 * fill in instead of staring at a "scanning…" message for 30 s.
 *
 * The threshold slider is purely client-side once patches arrive: each patch
 * carries its `maxDiff`, and we recolor green (kept) / red (skipped) without
 * re-running the scan. "Sort skipped first" clusters the noisy patches at
 * the top.
 *
 * Layout matches tunet.py:223 (the QWidget version): controls row +
 * scrollable thumbnail grid + stats counter.
 */

import { useEffect, useMemo, useRef, useState } from 'react'

interface Patch {
  x:         number
  y:         number
  src:       string
  maxDiff:   number
  thumb:     string  // base64 PNG — src patch
  diffThumb: string  // base64 PNG — grayscale |src-dst|, CSS filter amps it
}

interface PairInput {
  name: string
  src:  File
  dst:  File
}

interface PreviewFilterDialogProps {
  open:        boolean
  onClose:     () => void
  /** src↔dst File pairs (already matched by basename in the parent) */
  pairs:       PairInput[] | null
  resolution?: number
  overlap?:    number
  /** Initial threshold (matches the form's value) */
  initialThreshold?: number
  /** Called when the user accepts a new threshold */
  onAccept?:   (threshold: number) => void
}

interface ScanState {
  total:    number     // total candidate patches in the grid
  sampled:  number     // number we'll actually score (<= MAX_PATCHES)
  patches:  Patch[]    // streamed in as the worker scores them
  done:     boolean
}

export function PreviewFilterDialog({
  open, onClose, pairs,
  resolution = 512, overlap = 0.25,
  initialThreshold = 3.0,
  onAccept,
}: PreviewFilterDialogProps) {
  const [threshold, setThreshold] = useState(initialThreshold)
  const [size, setSize]           = useState(80)
  const [sortSkipped, setSort]    = useState(false)
  // Switch each tile from src thumbnail to a grayscale |src-dst| map. Useful
  // when the region of interest is small (smudge, dust, lens artifact) and
  // src tiles all look identical — the diff makes the differing region pop.
  const [showDiff, setShowDiff]   = useState(false)
  // CSS `filter: brightness(amp)` multiplier on the diff tile. 1× = raw,
  // higher = faint differences glow. Threshold math is unaffected — this is
  // pure display.
  const [amp, setAmp]             = useState(2.5)
  const [state, setState]         = useState<ScanState | null>(null)
  const [error, setError]         = useState<string | null>(null)
  const [scanning, setScanning]   = useState(false)
  const workerRef = useRef<Worker | null>(null)
  // Track what we last scanned so we don't redo work when the dialog reopens
  // unchanged (e.g. user closes and reopens to tweak threshold).
  const lastScanRef = useRef<{ pairsKey: string; resolution: number; overlap: number } | null>(null)

  // Spin up / tear down the worker alongside the dialog lifecycle. Each scan
  // gets a fresh worker so we never have to coordinate cancel-then-rescan
  // semantics inside the worker itself.
  useEffect(() => {
    if (!open || !pairs || pairs.length === 0) return

    const key = { pairsKey: pairsCacheKey(pairs), resolution, overlap }
    const prev = lastScanRef.current
    if (prev && prev.pairsKey === key.pairsKey && prev.resolution === key.resolution && prev.overlap === key.overlap) {
      // Already scanned this exact set — keep the existing results.
      return
    }
    lastScanRef.current = key

    setScanning(true)
    setError(null)
    setState({ total: 0, sampled: 0, patches: [], done: false })

    // NOTE: classic worker (no { type: 'module' }). Turbopack/webpack bundle
    // the worker into a single self-contained script that way; module workers
    // try to `importScripts()` chunked deps and fail in dev.
    const w = new Worker(new URL('@/lib/preview-filter-worker.ts', import.meta.url))
    workerRef.current = w

    w.onmessage = (e: MessageEvent) => {
      const msg = e.data
      switch (msg?.type) {
        case 'planned':
          setState(s => s ? { ...s, total: msg.total, sampled: msg.sampled } : s)
          break
        case 'patch':
          setState(s => s ? { ...s, patches: [...s.patches, msg.patch] } : s)
          break
        case 'done':
          setState(s => s ? { ...s, done: true } : s)
          setScanning(false)
          break
        case 'error':
          setError(msg.message ?? 'scan failed')
          setScanning(false)
          break
      }
    }
    w.onerror = (ev) => {
      setError(ev.message || 'worker error')
      setScanning(false)
    }

    w.postMessage({ type: 'scan', pairs, resolution, overlap })

    return () => {
      try { w.postMessage({ type: 'cancel' }) } catch {}
      w.terminate()
      workerRef.current = null
    }
    // pairs identity is stable between renders (provided by parent), so we
    // intentionally don't include it in deps to avoid restarting the scan
    // on every parent re-render. We re-key via lastScanRef + pairsCacheKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, resolution, overlap, pairs && pairsCacheKey(pairs)])

  const stats = useMemo(() => {
    if (!state) return null
    let kept = 0
    for (const p of state.patches) if (p.maxDiff >= threshold) kept++
    return { kept, skipped: state.patches.length - kept, scored: state.patches.length }
  }, [state, threshold])

  const ordered = useMemo(() => {
    if (!state) return [] as Patch[]
    if (!sortSkipped) return state.patches
    return [...state.patches].sort((a, b) => {
      const ak = a.maxDiff >= threshold ? 1 : 0
      const bk = b.maxDiff >= threshold ? 1 : 0
      if (ak !== bk) return ak - bk      // skipped (0) first
      return a.maxDiff - b.maxDiff
    })
  }, [state, sortSkipped, threshold])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-[#e5e7eb]">
          <h2 className="text-base font-semibold text-[#111827]">Preview Filter</h2>
          <button onClick={onClose} className="text-[#6b7280] hover:text-[#111827] text-xl leading-none">×</button>
        </div>

        {/* Controls */}
        <div className="px-5 py-3 border-b border-[#e5e7eb] flex items-center gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-[#374151]">Threshold:</span>
            <input
              // step="any" — step={0.5} from min={0.1} would reject e.g. 6.9
              // because Chrome enforces value = min + N*step.
              type="number" step="any" min={0.1} max={50}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value) || 0)}
              className="w-20 border border-[#e5e7eb] rounded px-2 py-1 text-sm font-mono"
            />
            <input
              type="range" min={0} max={30} step={0.1}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="accent-[#7E3AF2]"
              style={{ width: 200 }}
            />
          </label>

          <label className="flex items-center gap-2 text-sm">
            <span className="text-[#374151]">Size:</span>
            <input
              type="range" min={40} max={160} step={5}
              value={size}
              onChange={e => setSize(parseInt(e.target.value, 10))}
              className="accent-[#7E3AF2]"
              style={{ width: 120 }}
            />
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={sortSkipped}
              onChange={e => setSort(e.target.checked)}
              className="accent-[#7E3AF2]"
            />
            <span className="text-[#374151]">Sort skipped first</span>
          </label>

          <label
            className="flex items-center gap-2 text-sm"
            title="Display each tile as a grayscale |src - dst| map. Bright pixels mark where the source and target frames differ — use this to find patches that contain your region of interest (smudge, dust, etc.)."
          >
            <input
              type="checkbox"
              checked={showDiff}
              onChange={e => setShowDiff(e.target.checked)}
              className="accent-[#7E3AF2]"
            />
            <span className="text-[#374151]">Show diff</span>
          </label>

          {showDiff && (
            <label
              className="flex items-center gap-2 text-sm"
              title="Multiplier on diff brightness. 1× = raw, 2.5× = default, 5–10× for low-contrast regions, 20× saturates aggressively. Display-only — threshold math uses raw values."
            >
              <span className="text-[#374151]">Amp:</span>
              <input
                type="number" step={0.5} min={1} max={20}
                value={amp}
                onChange={e => setAmp(parseFloat(e.target.value) || 1)}
                className="w-16 border border-[#e5e7eb] rounded px-2 py-1 text-sm font-mono"
              />
              <input
                type="range" min={1} max={20} step={0.1}
                value={amp}
                onChange={e => setAmp(parseFloat(e.target.value))}
                className="accent-[#7E3AF2]"
                style={{ width: 120 }}
              />
            </label>
          )}

          <div className="ml-auto text-sm">
            {error    && <span className="text-[#EF4444]">Error: {error}</span>}
            {!error && scanning && state && state.sampled > 0 && (
              <span className="text-[#7E3AF2] font-mono text-xs">
                Scanning {state.patches.length} / {state.sampled}…
              </span>
            )}
            {!error && scanning && state && state.sampled === 0 && (
              <span className="text-[#7E3AF2] font-mono text-xs">Decoding…</span>
            )}
            {!error && stats && stats.scored > 0 && (
              <span className="font-mono text-xs">
                Scored: {stats.scored}{state && state.sampled > stats.scored ? ` / ${state.sampled}` : ''}{'  '}
                <span className="text-[#16A34A]">Kept: {stats.kept}</span>{'  '}
                <span className="text-[#EF4444]">Skipped: {stats.skipped}</span>{'  '}
                ({Math.round(100 * stats.skipped / Math.max(1, stats.scored))}% filtered)
              </span>
            )}
          </div>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-auto p-3 bg-[#fafafa]">
          {!state && !scanning && !error && (
            <div className="text-sm text-[#9ca3af] italic">Opening scan…</div>
          )}
          {scanning && state && state.patches.length === 0 && (
            <div className="text-sm text-[#9ca3af] italic">
              Decoding images locally — no upload needed. Patches will appear as they&apos;re scored.
            </div>
          )}
          {state && state.done && state.patches.length === 0 && (
            <div className="text-sm text-[#9ca3af] italic">No patches produced (images smaller than {resolution}px?).</div>
          )}
          {state && state.patches.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {ordered.map((p, i) => {
                const kept = p.maxDiff >= threshold
                const border = kept ? '#22c55e' : '#ef4444'
                return (
                  <div
                    key={i}
                    title={`${p.src} @ (${p.x},${p.y})  max diff: ${p.maxDiff.toFixed(1)}`}
                    style={{
                      width: size, height: size,
                      border: `${Math.max(2, Math.floor(size / 30))}px solid ${border}`,
                      boxSizing: 'border-box',
                      backgroundImage: `url(data:image/png;base64,${showDiff ? p.diffThumb : p.thumb})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                      // CSS brightness() is a linear channel multiplier per
                      // MDN — exactly what we want for amplifying the raw
                      // diff. Only applied on the diff thumb; src thumbs
                      // stay color-accurate.
                      filter: showDiff ? `brightness(${amp})` : undefined,
                      flex: '0 0 auto',
                    }}
                  />
                )
              })}
            </div>
          )}
          {state && state.done && state.sampled < state.total && (
            <p className="text-[11px] text-[#9ca3af] mt-3">
              Sampled {state.sampled} of {state.total} candidate patches.
              Threshold counts above are over the sampled set.
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-[#e5e7eb] flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-[#374151] hover:bg-[#f3f4f6] rounded"
          >
            Cancel
          </button>
          {onAccept && (
            <button
              onClick={() => { onAccept(threshold); onClose() }}
              className="px-4 py-1.5 text-sm bg-[#7E3AF2] hover:bg-[#6C2BD9] text-white rounded font-medium"
            >
              Use threshold {threshold.toFixed(1)}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ── helpers ─────────────────────────────────────────────────────────────────

/**
 * Fingerprint a pairs array so the scan-cache effect can detect "same
 * dataset" without re-scanning when the parent re-renders. File identity is
 * stable across renders within a single picker session, but a new folder
 * pick produces new File handles → different lastModified/size signature.
 */
function pairsCacheKey(pairs: PairInput[]): string {
  if (pairs.length === 0) return 'empty'
  // First + last + count is plenty to disambiguate folder picks; we don't
  // need a perfect hash, just something that flips when the user repicks.
  const first = pairs[0]
  const last  = pairs[pairs.length - 1]
  return `${pairs.length}|${first.src.name}:${first.src.size}:${first.src.lastModified}|${last.src.name}:${last.src.size}`
}
