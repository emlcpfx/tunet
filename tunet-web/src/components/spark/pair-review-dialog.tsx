'use client'

/**
 * Pair Review dialog — a pre-submit approval gate.
 *
 * Decodes every matched src↔dst pair in the browser (via pair-review-worker,
 * reading the picked folder's File handles — no upload) and shows each as a
 * Source | Target | Diff card. The user scans the grid to confirm the dataset
 * is the one they meant and that source frames map to the right targets, then
 * clicks "Approve" to unlock submission. New folder picks reset the approval
 * upstream (see new/page.tsx), so a wrong dataset can't slip through.
 *
 * Why this exists: a wrong/misaligned folder isn't caught by the pair-count or
 * Preview Filter checks, and a bad submission costs a full GPU training run.
 *
 * Mirrors the auto-mask / preview-filter dialog shell: worker lifecycle tied to
 * open state, results stream in, controls (thumb size + diff amp) are pure
 * client-side once decoded.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import type { PairOut } from '@/lib/pair-review-worker'

interface PairInput { name: string; src: File; dst: File }

interface PairReviewDialogProps {
  open:      boolean
  onClose:   () => void
  /** src↔dst File pairs (already matched by basename in the parent). */
  pairs:     PairInput[] | null
  /** Called when the user approves — the parent unlocks Submit. */
  onApprove: () => void
}

interface FailedPair { name: string; message: string }

const NEAR_IDENTICAL = 0.5   // meanDiff (0..255) below this ⇒ src≈dst

export function PairReviewDialog({ open, onClose, pairs, onApprove }: PairReviewDialogProps) {
  const [size, setSize]         = useState(120)
  const [amp, setAmp]           = useState(3)
  const [pairsOut, setPairsOut] = useState<PairOut[]>([])
  const [failed, setFailed]     = useState<FailedPair[]>([])
  const [total, setTotal]       = useState(0)
  const [scanning, setScanning] = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)
  // Avoid re-decoding when the dialog is reopened on the same dataset.
  const lastKeyRef = useRef<string | null>(null)

  useEffect(() => {
    if (!open || !pairs || pairs.length === 0) return

    const key = pairsKey(pairs)
    if (lastKeyRef.current === key && pairsOut.length > 0) return  // already decoded this set
    lastKeyRef.current = key

    setScanning(true)
    setError(null)
    setPairsOut([])
    setFailed([])
    setTotal(pairs.length)

    // Classic worker (no { type: 'module' }) — see pair-review-worker.ts header.
    const w = new Worker(new URL('@/lib/pair-review-worker.ts', import.meta.url))
    workerRef.current = w

    w.onmessage = (e: MessageEvent) => {
      const msg = e.data
      switch (msg?.type) {
        case 'planned':
          setTotal(msg.total)
          break
        case 'pair':
          setPairsOut(prev => [...prev, msg.pair as PairOut])
          break
        case 'pairError':
          setFailed(prev => [...prev, { name: msg.name, message: msg.message }])
          break
        case 'done':
          setScanning(false)
          break
        case 'error':
          setError(msg.message ?? 'decode failed')
          setScanning(false)
          break
      }
    }
    w.onerror = (ev) => { setError(ev.message || 'worker error'); setScanning(false) }

    w.postMessage({ type: 'scan', pairs })

    return () => {
      try { w.postMessage({ type: 'cancel' }) } catch { /* noop */ }
      w.terminate()
      workerRef.current = null
    }
    // pairs identity is stable per the parent's useMemo; re-key via lastKeyRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, pairs && pairsKey(pairs)])

  const mismatches = useMemo(() => pairsOut.filter(p => p.dimsMismatch), [pairsOut])
  const identicals = useMemo(
    () => pairsOut.filter(p => !p.dimsMismatch && p.meanDiff < NEAR_IDENTICAL),
    [pairsOut],
  )

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-[#e5e7eb]">
          <div>
            <h2 className="text-base font-semibold text-[#111827]">Review Pairs</h2>
            <p className="text-xs text-[#6b7280]">
              Confirm this is the right dataset and each source maps to the correct target before training.
            </p>
          </div>
          <button onClick={onClose} className="text-[#6b7280] hover:text-[#111827] text-xl leading-none">×</button>
        </div>

        {/* Controls */}
        <div className="px-5 py-3 border-b border-[#e5e7eb] flex items-center gap-5 flex-wrap">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-[#374151]">Size:</span>
            <input
              type="range" min={64} max={240} step={4}
              value={size}
              onChange={e => setSize(parseInt(e.target.value, 10))}
              className="accent-[#7E3AF2]"
              style={{ width: 120 }}
            />
          </label>

          <label className="flex items-center gap-2 text-sm" title="Brighten the diff column so subtle source→target changes are visible. Display-only.">
            <span className="text-[#374151]">Diff amp:</span>
            <input
              type="range" min={1} max={20} step={0.5}
              value={amp}
              onChange={e => setAmp(parseFloat(e.target.value))}
              className="accent-[#7E3AF2]"
              style={{ width: 120 }}
            />
            <span className="font-mono text-xs text-[#9ca3af] w-8">{amp.toFixed(1)}×</span>
          </label>

          <div className="ml-auto text-xs flex items-center gap-4">
            {error && <span className="text-[#EF4444]">Error: {error}</span>}
            {!error && (
              <span className="font-mono text-[#9ca3af]">
                {scanning ? `Decoding ${pairsOut.length} / ${total}…` : `${pairsOut.length} pairs`}
              </span>
            )}
            {mismatches.length > 0 && (
              <span className="text-[#EF4444] font-semibold" title="Source and target differ in resolution — likely a wrong or misaligned pairing.">
                ⚠ {mismatches.length} mismatched dims
              </span>
            )}
            {identicals.length > 0 && (
              <span className="text-[#D97706]" title="Source and target are nearly identical — check you didn't point both folders at the same place.">
                ◦ {identicals.length} identical
              </span>
            )}
          </div>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-auto p-3 bg-[#fafafa]">
          {pairsOut.length === 0 && scanning && (
            <div className="text-sm text-[#9ca3af] italic">
              Decoding image pairs locally — no upload needed. Cards appear as they&apos;re decoded.
            </div>
          )}
          {failed.length > 0 && (
            <div className="mb-3 px-3 py-2 bg-[#FEF2F2] border border-[#fecaca] rounded text-xs text-[#7F1D1D]">
              <p className="font-semibold text-[#EF4444]">{failed.length} pair{failed.length === 1 ? '' : 's'} failed to decode:</p>
              <ul className="mt-1 max-h-24 overflow-auto list-disc pl-5 font-mono">
                {failed.slice(0, 50).map(f => <li key={f.name}>{f.name} — {f.message}</li>)}
              </ul>
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            {pairsOut.map((p, i) => (
              <PairCard key={`${p.name}-${i}`} pair={p} size={size} amp={amp} />
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-[#e5e7eb] flex items-center justify-end gap-3">
          {mismatches.length > 0 && (
            <span className="mr-auto text-xs text-[#EF4444]">
              ⚠ {mismatches.length} pair{mismatches.length === 1 ? '' : 's'} have mismatched dimensions — double-check before approving.
            </span>
          )}
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-[#374151] hover:bg-[#f3f4f6] rounded"
          >
            Cancel
          </button>
          <button
            onClick={() => { onApprove(); onClose() }}
            // Disabled only while actively decoding with nothing shown yet.
            // Once decoding finishes we enable it even if every pair failed to
            // decode (e.g. TIFF, which the browser can't render) — otherwise a
            // decode limitation would soft-lock the user out of submitting.
            disabled={scanning && pairsOut.length === 0}
            className="px-4 py-1.5 text-sm bg-[#7E3AF2] hover:bg-[#6C2BD9] disabled:bg-[#c4b5fd] disabled:cursor-not-allowed text-white rounded font-medium"
          >
            {pairsOut.length > 0 ? `Approve ${pairsOut.length} pairs` : 'Approve anyway'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

function PairCard({ pair, size, amp }: { pair: PairOut; size: number; amp: number }) {
  const ring =
    pair.dimsMismatch                          ? 'border-[#EF4444]' :
    pair.meanDiff < NEAR_IDENTICAL             ? 'border-[#D97706]' :
                                                 'border-[#e5e7eb]'
  return (
    <div className={`bg-white border ${ring} rounded-md overflow-hidden`} style={{ borderWidth: 1.5 }}>
      <div className="px-2 py-1 border-b border-[#e5e7eb] flex items-center justify-between gap-2" style={{ maxWidth: size * 3 }}>
        <span className="font-mono text-[11px] text-[#374151] truncate" title={pair.name}>{pair.name}</span>
        {pair.dimsMismatch ? (
          <span className="text-[10px] text-[#EF4444] font-semibold whitespace-nowrap" title={`src ${pair.srcW}×${pair.srcH} vs dst ${pair.dstW}×${pair.dstH}`}>
            {pair.srcW}×{pair.srcH} ≠ {pair.dstW}×{pair.dstH}
          </span>
        ) : pair.meanDiff < NEAR_IDENTICAL ? (
          <span className="text-[10px] text-[#D97706] whitespace-nowrap">identical</span>
        ) : null}
      </div>
      <div className="flex gap-px bg-[#e5e7eb]">
        <ImgCell label="Source" src={pair.srcThumb} size={size} />
        <ImgCell label="Target" src={pair.dstThumb} size={size} />
        {pair.diffThumb ? (
          <ImgCell label="Diff" src={pair.diffThumb} size={size} brightness={amp} />
        ) : (
          <div className="bg-white flex flex-col" style={{ width: size }}>
            <div className="px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-[#9ca3af] font-semibold">Diff</div>
            <div className="flex-1 flex items-center justify-center text-[10px] text-[#9ca3af] italic px-1 text-center" style={{ minHeight: size * 0.6 }}>
              dims differ
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ImgCell({ label, src, size, brightness }: {
  label: string; src: string; size: number; brightness?: number
}) {
  return (
    <div className="bg-white" style={{ width: size }}>
      <div className="px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-[#9ca3af] font-semibold">{label}</div>
      {/* eslint-disable-next-line @next/next/no-img-element -- base64 data URI, not a remote asset */}
      <img
        src={`data:image/png;base64,${src}`}
        alt={label}
        style={{
          width: size, height: 'auto', display: 'block',
          filter: brightness ? `brightness(${brightness})` : undefined,
        }}
      />
    </div>
  )
}

/** Fingerprint a pairs array so the decode effect can skip an unchanged reopen. */
function pairsKey(pairs: PairInput[]): string {
  if (pairs.length === 0) return 'empty'
  const first = pairs[0]
  const last  = pairs[pairs.length - 1]
  return `${pairs.length}|${first.src.name}:${first.src.size}:${first.src.lastModified}|${last.src.name}:${last.src.size}`
}
