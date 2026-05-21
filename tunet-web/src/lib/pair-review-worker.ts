/// <reference lib="webworker" />

/**
 * Pair-review Web Worker — decodes EVERY matched src↔dst pair from the picked
 * folder (local File handles, no upload) and emits a Source thumbnail, a Target
 * thumbnail, and a grayscale |src−dst| diff thumbnail for each. Powers the
 * pre-submit "Review Pairs" approval gate so a wrong or misaligned dataset is
 * caught before a training run is paid for.
 *
 * Mirrors preview-filter-worker.ts's decode path (createImageBitmap for common
 * formats, parse-exr + exrFloatRgbaToRgb8 for EXR) but operates on whole frames
 * downscaled to a thumbnail rather than scoring patches. Eager: it loops every
 * pair and posts each result as it lands so the grid fills in progressively.
 *
 * Wire protocol (postMessage):
 *   in:  { type: 'scan', pairs: PairInput[], thumbMaxEdge? }
 *   in:  { type: 'cancel' }
 *   out: { type: 'planned',   total }
 *   out: { type: 'pair',      index, total, pair: PairOut }
 *   out: { type: 'pairError', index, total, name, message }
 *   out: { type: 'done',      emitted }
 *   out: { type: 'error',     message }
 *
 * NOTE: classic worker (no { type: 'module' }) — same reason as
 * preview-filter-worker.ts: Turbopack/webpack bundle parse-exr's deps into one
 * self-contained script that way; module workers try importScripts() and fail.
 */

import parseExr from 'parse-exr'
import { exrFloatRgbaToRgb8 } from './preview-filter-core'

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope

export interface PairInput { name: string; src: File; dst: File }

export interface PairOut {
  name:        string
  srcThumb:    string        // base64 PNG (srcTw × srcTh)
  srcTw:       number
  srcTh:       number
  dstThumb:    string        // base64 PNG (dstTw × dstTh)
  dstTw:       number
  dstTh:       number
  diffThumb:   string | null // base64 PNG (srcTw × srcTh); null when dims mismatch
  /** Full-frame dimensions, so the UI can surface a mismatch precisely. */
  srcW: number; srcH: number
  dstW: number; dstH: number
  /** src and dst differ in resolution — almost always a wrong/misaligned pair. */
  dimsMismatch: boolean
  /** Mean channel-abs-diff over the thumb, 0..255. ~0 ⇒ src and dst are identical. */
  meanDiff:    number
}

interface ScanMsg   { type: 'scan'; pairs: PairInput[]; thumbMaxEdge?: number }
interface CancelMsg { type: 'cancel' }
type InMsg = ScanMsg | CancelMsg

interface FullImage { canvas: OffscreenCanvas; width: number; height: number }

let cancelled = false

ctx.addEventListener('message', (e: MessageEvent<InMsg>) => {
  const msg = e.data
  if (msg.type === 'cancel') { cancelled = true; return }
  if (msg.type === 'scan') {
    cancelled = false
    runScan(msg).catch(err => post({ type: 'error', message: err?.message ?? String(err) }))
  }
})

async function runScan(msg: ScanMsg) {
  const { pairs } = msg
  const maxEdge = msg.thumbMaxEdge ?? 224
  if (pairs.length === 0) {
    post({ type: 'error', message: 'no matched src↔dst pairs' })
    return
  }

  post({ type: 'planned', total: pairs.length })

  let emitted = 0
  for (let i = 0; i < pairs.length; i++) {
    if (cancelled) return
    try {
      const sFull = await decodeToFullCanvas(pairs[i].src)
      const dFull = await decodeToFullCanvas(pairs[i].dst)

      // Each side downscales to its OWN aspect-correct thumb. For matched
      // dimensions (the norm) the two thumbs are pixel-identical in size, so a
      // diff is well-defined; a mismatch renders both at their true shapes,
      // making the difference visually obvious, and the diff is skipped.
      const s = renderThumb(sFull, maxEdge)
      const d = renderThumb(dFull, maxEdge)

      const dimsMismatch = sFull.width !== dFull.width || sFull.height !== dFull.height

      let diffThumb: string | null = null
      let meanDiff = 0
      if (!dimsMismatch && s.tw === d.tw && s.th === d.th) {
        const diff = diffRgba(s.rgba, d.rgba, s.tw, s.th)
        diffThumb = await rgbaToPng(diff.rgba, s.tw, s.th)
        meanDiff  = diff.mean
      }

      const pair: PairOut = {
        name:     pairs[i].name,
        srcThumb: await rgbaToPng(s.rgba, s.tw, s.th), srcTw: s.tw, srcTh: s.th,
        dstThumb: await rgbaToPng(d.rgba, d.tw, d.th), dstTw: d.tw, dstTh: d.th,
        diffThumb,
        srcW: sFull.width, srcH: sFull.height,
        dstW: dFull.width, dstH: dFull.height,
        dimsMismatch, meanDiff,
      }
      post({ type: 'pair', index: i, total: pairs.length, pair })
      emitted++
    } catch (err) {
      // Surface the failed filename rather than aborting the whole review.
      post({
        type: 'pairError', index: i, total: pairs.length,
        name: pairs[i].name, message: (err as Error)?.message ?? 'decode failed',
      })
    }
  }

  post({ type: 'done', emitted })
}

// ── Decode ───────────────────────────────────────────────────────────────────

/** Decode any supported image to a full-resolution OffscreenCanvas. */
async function decodeToFullCanvas(file: File): Promise<FullImage> {
  if (extName(file.name) === '.exr') {
    const ab  = await file.arrayBuffer()
    const exr = parseExr(ab, 1015)            // Float32 — matches the other workers
    const w   = exr.width
    const h   = exr.height
    const rgb = exrFloatRgbaToRgb8(exr.data as Float32Array, w, h)  // already vertically un-flipped
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      rgba[j] = rgb[i]; rgba[j + 1] = rgb[i + 1]; rgba[j + 2] = rgb[i + 2]; rgba[j + 3] = 255
    }
    const canvas = new OffscreenCanvas(w, h)
    const c2d = canvas.getContext('2d')
    if (!c2d) throw new Error('OffscreenCanvas 2D context unavailable')
    c2d.putImageData(new ImageData(rgba, w, h), 0, 0)
    return { canvas, width: w, height: h }
  }

  const bmp = await createImageBitmap(file)
  const w = bmp.width
  const h = bmp.height
  const canvas = new OffscreenCanvas(w, h)
  const c2d = canvas.getContext('2d')
  if (!c2d) { bmp.close(); throw new Error('OffscreenCanvas 2D context unavailable') }
  c2d.drawImage(bmp, 0, 0)
  bmp.close()
  return { canvas, width: w, height: h }
}

/** Downscale a full image into an aspect-correct thumb; returns its RGBA + dims. */
function renderThumb(full: FullImage, maxEdge: number): { rgba: Uint8ClampedArray; tw: number; th: number } {
  const scale = Math.min(1, maxEdge / Math.max(full.width, full.height))
  const tw = Math.max(1, Math.round(full.width  * scale))
  const th = Math.max(1, Math.round(full.height * scale))
  const canvas = new OffscreenCanvas(tw, th)
  const c2d = canvas.getContext('2d', { willReadFrequently: true })
  if (!c2d) throw new Error('OffscreenCanvas 2D context unavailable')
  c2d.drawImage(full.canvas, 0, 0, tw, th)
  return { rgba: c2d.getImageData(0, 0, tw, th).data, tw, th }
}

// ── Diff + encode ──────────────────────────────────────────────────────────────

/**
 * Grayscale channel-mean |src − dst| over equal-sized RGBA thumbs. Matches the
 * metric used by the Preview Filter (preview-filter-core.patchMaxAbsDiff), so
 * the two tools agree on what "different" means.
 */
function diffRgba(
  a: Uint8ClampedArray, b: Uint8ClampedArray, w: number, h: number,
): { rgba: Uint8ClampedArray; mean: number } {
  const out = new Uint8ClampedArray(w * h * 4)
  let sum = 0
  for (let i = 0; i < a.length; i += 4) {
    const dr = Math.abs(a[i]     - b[i])
    const dg = Math.abs(a[i + 1] - b[i + 1])
    const db = Math.abs(a[i + 2] - b[i + 2])
    const g  = (dr + dg + db) / 3
    sum += g
    out[i] = g; out[i + 1] = g; out[i + 2] = g; out[i + 3] = 255
  }
  return { rgba: out, mean: sum / (w * h) }
}

async function rgbaToPng(rgba: Uint8ClampedArray, w: number, h: number): Promise<string> {
  const canvas = new OffscreenCanvas(w, h)
  const c2d = canvas.getContext('2d')
  if (!c2d) throw new Error('OffscreenCanvas 2D context unavailable')
  // createImageData + set rather than `new ImageData(rgba, …)`: the source may
  // be a getImageData()-derived buffer (typed ArrayBufferLike), which the
  // ImageData constructor's ArrayBuffer-only signature rejects.
  const img = c2d.createImageData(w, h)
  img.data.set(rgba)
  c2d.putImageData(img, 0, 0)
  const blob = await canvas.convertToBlob({ type: 'image/png' })
  return arrayBufferToBase64(await blob.arrayBuffer())
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let bin = ''
  const CHUNK = 0x8000
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)))
  }
  return btoa(bin)
}

function extName(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot < 0 ? '' : name.slice(dot).toLowerCase()
}

function post(msg: unknown) {
  ctx.postMessage(msg)
}
