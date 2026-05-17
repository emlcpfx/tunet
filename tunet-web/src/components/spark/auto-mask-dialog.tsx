'use client'

/**
 * Auto-Mask Preview dialog.
 *
 * Runs auto-mask-worker.ts on a small sample of src/dst pairs to produce the
 * pre-gamma mask intensity. Then re-applies `mask^gamma` on the main thread
 * every time the user moves the slider, so the preview is live without
 * re-decoding or re-blurring.
 *
 * Layout: header with gamma slider, body shows N samples side-by-side
 *   [src thumb] | [mask at current gamma] | [src ⊙ mask overlay]
 *
 * Pairs are passed in from the parent (same shape the Preview Filter uses).
 * We only sample MAX_SAMPLES of them — past 4-6 the dialog gets noisy and
 * the user can't actually compare.
 */

import { useEffect, useMemo, useRef, useState } from 'react'

interface PairInput { name: string; src: File; dst: File }

interface Sample {
  name:   string
  srcPng: string         // base64 PNG of the downsampled source
  mask:   Uint8Array     // pre-gamma intensity, length = size²
  size:   number
  empty:  boolean
}

interface AutoMaskDialogProps {
  open:        boolean
  onClose:     () => void
  pairs:       PairInput[] | null
  /** Initial gamma value (mirrors the form's auto_mask_gamma). */
  initialGamma?: number
  /** Called when the user clicks "Use this gamma". */
  onAccept?:   (gamma: number) => void
}

const MAX_SAMPLES  = 4
const PREVIEW_SIZE = 256

export function AutoMaskDialog({
  open, onClose, pairs,
  initialGamma = 1.0,
  onAccept,
}: AutoMaskDialogProps) {
  const [gamma, setGamma]       = useState(initialGamma)
  const [samples, setSamples]   = useState<Sample[]>([])
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 })
  const [computing, setComputing] = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)
  // Track what we last computed so we don't redo work when the dialog reopens
  // unchanged (e.g. user closes and reopens to tweak gamma).
  const lastKeyRef = useRef<string | null>(null)

  // Reset gamma when dialog opens with a new initial.
  useEffect(() => {
    if (open) setGamma(initialGamma)
  }, [open, initialGamma])

  // Spin up the worker when the dialog opens with a fresh dataset.
  useEffect(() => {
    if (!open || !pairs || pairs.length === 0) return

    const sliced = pairs.slice(0, MAX_SAMPLES)
    const key    = sliced.map(p => `${p.name}:${p.src.size}:${p.dst.size}`).join('|')
    if (lastKeyRef.current === key && samples.length > 0) {
      return  // already computed this exact set
    }
    lastKeyRef.current = key

    setComputing(true)
    setError(null)
    setSamples([])
    setProgress({ done: 0, total: sliced.length })

    const w = new Worker(new URL('@/lib/auto-mask-worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = w

    w.onmessage = (e: MessageEvent) => {
      const msg = e.data
      switch (msg?.type) {
        case 'sample':
          setSamples(prev => [...prev, msg.sample as Sample])
          setProgress({ done: msg.index + 1, total: msg.total })
          break
        case 'done':
          setComputing(false)
          break
        case 'error':
          setError(msg.message ?? 'compute failed')
          setComputing(false)
          break
      }
    }
    w.onerror = (ev) => {
      setError(ev.message || 'worker error')
      setComputing(false)
    }

    w.postMessage({ type: 'compute', pairs: sliced, previewSize: PREVIEW_SIZE })

    return () => {
      try { w.postMessage({ type: 'cancel' }) } catch { /* noop */ }
      w.terminate()
      workerRef.current = null
    }
    // pairs identity is stable per parent's useMemo; we re-key via lastKeyRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, pairs])

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
          <div>
            <h2 className="text-base font-semibold text-[#111827]">Auto-Mask Preview</h2>
            <p className="text-xs text-[#6b7280]">
              White areas weight loss higher during training. Adjust gamma to control how aggressively the mask spreads.
            </p>
          </div>
          <button onClick={onClose} className="text-[#6b7280] hover:text-[#111827] text-xl leading-none">×</button>
        </div>

        {/* Gamma slider */}
        <div className="px-5 py-3 border-b border-[#e5e7eb] flex items-center gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-[#374151]">Gamma:</span>
            <input
              // step="any" — step={0.05} from min={0.1} restricts typing to
              // 0.10, 0.15, 0.20… which rejects values like 0.13 or 0.77.
              type="number" step="any" min={0.1} max={5}
              value={gamma}
              onChange={e => setGamma(parseFloat(e.target.value) || 0.1)}
              className="w-20 border border-[#e5e7eb] rounded px-2 py-1 text-sm font-mono"
            />
            <input
              type="range" min={0.1} max={3.0} step={0.05}
              value={gamma}
              onChange={e => setGamma(parseFloat(e.target.value))}
              className="accent-[#7E3AF2]"
              style={{ width: 240 }}
            />
          </label>

          <div className="flex items-center gap-2 text-xs">
            <button
              type="button"
              onClick={() => setGamma(0.5)}
              className="px-2 py-0.5 rounded border border-[#e5e7eb] hover:bg-[#F9FAFB] font-mono"
              title="0.5 — beauty preset default"
            >0.5</button>
            <button
              type="button"
              onClick={() => setGamma(1.0)}
              className="px-2 py-0.5 rounded border border-[#e5e7eb] hover:bg-[#F9FAFB] font-mono"
              title="1.0 — no curve, raw mask"
            >1.0</button>
            <button
              type="button"
              onClick={() => setGamma(2.0)}
              className="px-2 py-0.5 rounded border border-[#e5e7eb] hover:bg-[#F9FAFB] font-mono"
              title="2.0 — tight focus, most pixels become black"
            >2.0</button>
          </div>

          <div className="ml-auto text-xs">
            {computing && (
              <span className="text-[#7E3AF2]">
                Computing… {progress.done}/{progress.total}
              </span>
            )}
            {error && <span className="text-[#EF4444]">Error: {error}</span>}
            {!computing && !error && samples.length > 0 && (
              <span className="font-mono text-[#9ca3af]">
                {samples.length} sample{samples.length === 1 ? '' : 's'}
              </span>
            )}
          </div>
        </div>

        {/* Sample grid */}
        <div className="flex-1 overflow-auto p-4 bg-[#fafafa] space-y-3">
          {samples.length === 0 && computing && (
            <div className="text-sm text-[#9ca3af] italic">Decoding sample images…</div>
          )}
          {samples.length === 0 && !computing && !error && (
            <div className="text-sm text-[#9ca3af] italic">No samples yet.</div>
          )}
          {samples.map((s, i) => (
            <SampleRow key={i} sample={s} gamma={gamma} />
          ))}
          <p className="text-[11px] text-[#9ca3af] mt-3">
            Mask pipeline: <code>blur(|src−dst|)</code> → normalize → <code>sigmoid(20·(x−0.15))</code> → renormalize → <code>x<sup>γ</sup></code>.
            Lowering γ expands the white areas; raising γ tightens the focus.
            Empty rows mean the source and target were nearly identical (mask treated as zero — no training signal there).
          </p>
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
              onClick={() => { onAccept(gamma); onClose() }}
              className="px-4 py-1.5 text-sm bg-[#7E3AF2] hover:bg-[#6C2BD9] text-white rounded font-medium"
            >
              Use γ = {gamma.toFixed(2)}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Render one sample row: source thumbnail, mask at current gamma, and a
 * ½/½ overlay where the mask gates the source. The mask render is pure
 * canvas pixel work — gamma changes don't trigger any worker message, just
 * a re-paint of the two derived canvases.
 */
function SampleRow({ sample, gamma }: { sample: Sample; gamma: number }) {
  const maskRef    = useRef<HTMLCanvasElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const [coverage, setCoverage] = useState<number | null>(null)

  // Repaint mask + overlay whenever gamma changes.
  useEffect(() => {
    if (sample.empty) {
      paintEmpty(maskRef.current, sample.size)
      paintEmpty(overlayRef.current, sample.size)
      setCoverage(0)
      return
    }
    const cov = paintMask(maskRef.current, sample.mask, sample.size, gamma)
    setCoverage(cov)
    paintOverlay(overlayRef.current, sample.srcPng, sample.mask, sample.size, gamma)
  }, [sample, gamma])

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-md overflow-hidden">
      <div className="px-3 py-2 border-b border-[#e5e7eb] flex items-center justify-between">
        <span className="font-mono text-xs text-[#374151] truncate">{sample.name}</span>
        <span className="text-[11px] text-[#9ca3af]">
          {sample.empty
            ? 'Empty (no diff above noise threshold)'
            : coverage != null ? `Mask coverage: ${(coverage * 100).toFixed(1)}%` : ''}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-px bg-[#e5e7eb]">
        <ImageCell label="Source" src={`data:image/png;base64,${sample.srcPng}`} size={sample.size} />
        <CanvasCell label="Mask" canvasRef={maskRef} size={sample.size} dark />
        <CanvasCell label="Source × mask" canvasRef={overlayRef} size={sample.size} dark />
      </div>
    </div>
  )
}

function ImageCell({ label, src, size }: { label: string; src: string; size: number }) {
  return (
    <div className="bg-white">
      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">{label}</div>
      <img src={src} alt={label} width={size} height={size}
           className="block w-full h-auto" style={{ imageRendering: 'auto' }} />
    </div>
  )
}

function CanvasCell({
  label, canvasRef, size, dark,
}: {
  label: string
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  size: number
  dark?: boolean
}) {
  return (
    <div className={dark ? 'bg-black' : 'bg-white'}>
      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold bg-white">{label}</div>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="block w-full h-auto"
        style={{ imageRendering: 'auto' }}
      />
    </div>
  )
}

// ── Pure-canvas painters ────────────────────────────────────────────────────

function paintMask(
  canvas: HTMLCanvasElement | null, mask: Uint8Array, size: number, gamma: number,
): number | null {
  if (!canvas) return null
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  const img = ctx.createImageData(size, size)
  let coverageSum = 0
  // Apply x^gamma (where x is mask/255) and write to RGBA.
  const invMaxByte = 1 / 255
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] === 0 ? 0 : Math.pow(mask[i] * invMaxByte, gamma)
    const b = Math.max(0, Math.min(255, Math.round(v * 255)))
    const j = i * 4
    img.data[j]     = b
    img.data[j + 1] = b
    img.data[j + 2] = b
    img.data[j + 3] = 255
    coverageSum += v
  }
  ctx.putImageData(img, 0, 0)
  return coverageSum / mask.length
}

function paintOverlay(
  canvas: HTMLCanvasElement | null, srcPngB64: string,
  mask: Uint8Array, size: number, gamma: number,
) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Decode the source PNG once per gamma change. Cheap because it's already
  // in browser cache (data URI) and only 256x256.
  const im = new Image()
  im.onload = () => {
    ctx.drawImage(im, 0, 0, size, size)
    const data = ctx.getImageData(0, 0, size, size)
    const invMaxByte = 1 / 255
    for (let i = 0; i < mask.length; i++) {
      const v = mask[i] === 0 ? 0 : Math.pow(mask[i] * invMaxByte, gamma)
      const j = i * 4
      data.data[j]     = Math.round(data.data[j]     * v)
      data.data[j + 1] = Math.round(data.data[j + 1] * v)
      data.data[j + 2] = Math.round(data.data[j + 2] * v)
    }
    ctx.putImageData(data, 0, 0)
  }
  im.src = `data:image/png;base64,${srcPngB64}`
}

function paintEmpty(canvas: HTMLCanvasElement | null, size: number) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, size, size)
}
