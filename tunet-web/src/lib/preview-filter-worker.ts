/// <reference lib="webworker" />

/**
 * Preview-filter Web Worker — runs the patch-skip scan entirely in the user's
 * browser, reading the File handles the folder picker already gave us. The
 * old version uploaded a staged copy to the server and decoded with sharp;
 * this one decodes with createImageBitmap (and parse-exr for EXR), keeping
 * the bytes on the client.
 *
 * Wire protocol (postMessage):
 *
 *   in:  { type: 'scan', pairs: PairInput[], resolution, overlap }
 *   in:  { type: 'cancel' }
 *
 *   out: { type: 'pairs',   matched: number, totalSrc, totalDst }
 *   out: { type: 'planned', total: number, sampled: number, stride }
 *   out: { type: 'patch',   patch: PatchOut, index, total }
 *   out: { type: 'done' }
 *   out: { type: 'error',   message }
 *
 * The patch metric stays byte-identical with tunet.py:302 (channel-mean of
 * |delta|, max over pixels) — see preview-filter-core.ts.
 */

import parseExr from 'parse-exr'
import {
  exrFloatRgbaToRgb8,
  patchMaxAbsDiff,
  planPatches,
  sampleEvenly,
  MAX_PATCHES,
  SCAN_THUMB_SIZE,
} from './preview-filter-core'

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope

export interface PairInput {
  /** Original filename for display (e.g. 'frame_0001.exr') */
  name:    string
  src:     File
  dst:     File
}

export interface PatchOut {
  x:        number
  y:        number
  src:      string   // file basename
  maxDiff:  number
  thumb:    string   // base64 PNG — src patch downscaled to SCAN_THUMB_SIZE
  diffThumb: string  // base64 PNG — grayscale |src-dst| at 1× amp; dialog
                     //              applies CSS filter brightness(amp) on render
}

interface ScanMsg   { type: 'scan'; pairs: PairInput[]; resolution: number; overlap: number }
interface CancelMsg { type: 'cancel' }
type InMsg = ScanMsg | CancelMsg

interface DecodedRgb8 { data: Uint8Array; width: number; height: number }

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
  const { pairs, resolution, overlap } = msg
  if (pairs.length === 0) {
    post({ type: 'error', message: 'no matched src↔dst pairs' })
    return
  }

  post({ type: 'pairs', matched: pairs.length })

  // ── Phase 1: probe dims for every pair so we can plan the patch grid.
  // EXR has no cheap dim probe (parse-exr decodes the whole file), so we
  // reuse the decoded bytes via decodeCache instead of throwing them away.
  const decodeCache = new Map<number, DecodedRgb8>()  // key: pairIdx*2 + (0|1)  (0=src, 1=dst)
  const dims: ({ width: number; height: number } | null)[] = []
  for (let i = 0; i < pairs.length; i++) {
    if (cancelled) return
    try {
      const d = await readDimsCached(pairs[i].src, i * 2 + 0, decodeCache)
      dims.push(d)
    } catch {
      dims.push(null)
    }
  }

  // ── Phase 2: plan + sample the patches.
  const compactDims = dims.map(d => d ?? { width: 0, height: 0 })
  const { plans, stride } = planPatches(compactDims, resolution, overlap)

  if (plans.length === 0) {
    post({ type: 'error', message: 'no valid patches (images too small?)' })
    return
  }

  const sampledPlans = plans.length <= MAX_PATCHES ? plans : sampleEvenly(plans, MAX_PATCHES)
  post({ type: 'planned', total: plans.length, sampled: sampledPlans.length, stride })

  // ── Phase 3: per-patch decode + diff. Reuse decodeCache so each file is
  // decoded at most twice (once src, once dst) regardless of patch count.
  let emitted = 0
  for (let i = 0; i < sampledPlans.length; i++) {
    if (cancelled) return
    const plan = sampledPlans[i]
    try {
      const sImg = await decodePairSide(pairs, plan.fileIdx, 0, decodeCache)
      const dImg = await decodePairSide(pairs, plan.fileIdx, 1, decodeCache)
      if (sImg.width !== dImg.width || sImg.height !== dImg.height) continue

      const maxDiff = patchMaxAbsDiff(sImg.data, dImg.data, sImg.width,
        plan.x, plan.y, plan.w, plan.h)

      const thumb     = await renderThumb(sImg, plan.x, plan.y, plan.w, plan.h)
      const diffThumb = await renderDiffThumb(sImg, dImg, plan.x, plan.y, plan.w, plan.h)

      post({
        type: 'patch',
        index: i,
        total: sampledPlans.length,
        patch: {
          x:         plan.x,
          y:         plan.y,
          src:       pairs[plan.fileIdx].name,
          maxDiff,
          thumb,
          diffThumb,
        },
      })
      emitted++
    } catch {
      // skip patch on decode/extract error — we still want to surface as many
      // patches as possible rather than aborting the whole scan
    }
  }

  post({ type: 'done', emitted })
}

async function readDimsCached(
  file: File, cacheKey: number, cache: Map<number, DecodedRgb8>,
): Promise<{ width: number; height: number }> {
  const ext = extName(file.name)
  if (ext === '.exr') {
    // EXR has no cheap dim probe — decode now, keep in cache for the patch loop.
    const decoded = await decodeExr(file)
    cache.set(cacheKey, decoded)
    return { width: decoded.width, height: decoded.height }
  }
  // Common formats: createImageBitmap is the cheapest path. We don't cache
  // bitmaps because they each pin a full GPU/CPU buffer and the patch loop
  // re-decodes lazily into the same cache.
  const bmp = await createImageBitmap(file)
  const w = bmp.width
  const h = bmp.height
  bmp.close()
  return { width: w, height: h }
}

async function decodePairSide(
  pairs: PairInput[], pairIdx: number, side: 0 | 1,
  cache: Map<number, DecodedRgb8>,
): Promise<DecodedRgb8> {
  const key = pairIdx * 2 + side
  const cached = cache.get(key)
  if (cached) return cached
  const file = side === 0 ? pairs[pairIdx].src : pairs[pairIdx].dst
  const out = extName(file.name) === '.exr'
    ? await decodeExr(file)
    : await decodeBitmap(file)
  cache.set(key, out)
  return out
}

async function decodeBitmap(file: File): Promise<DecodedRgb8> {
  const bmp = await createImageBitmap(file)
  const w = bmp.width
  const h = bmp.height
  // OffscreenCanvas + getImageData is the only way to get raw pixels in a
  // worker — there's no `<img>` element here.
  const canvas = new OffscreenCanvas(w, h)
  const gctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!gctx) {
    bmp.close()
    throw new Error('OffscreenCanvas 2D context unavailable')
  }
  gctx.drawImage(bmp, 0, 0)
  bmp.close()
  const img = gctx.getImageData(0, 0, w, h)
  // ImageData is RGBA — repack to RGB.
  const out = new Uint8Array(w * h * 3)
  const src = img.data
  for (let i = 0, j = 0; i < src.length; i += 4, j += 3) {
    out[j]     = src[i]
    out[j + 1] = src[i + 1]
    out[j + 2] = src[i + 2]
  }
  return { data: out, width: w, height: h }
}

async function decodeExr(file: File): Promise<DecodedRgb8> {
  const ab = await file.arrayBuffer()
  const exr = parseExr(ab, 1015)  // Float32 mode — matches the server route
  const w = exr.width
  const h = exr.height
  const data = exrFloatRgbaToRgb8(exr.data as Float32Array, w, h)
  return { data, width: w, height: h }
}

/**
 * Extract `[x..x+w, y..y+h]` from a full-frame RGB byte buffer, resize to
 * SCAN_THUMB_SIZE × SCAN_THUMB_SIZE on an OffscreenCanvas, encode as PNG, and
 * return a base64 string (same wire format as the server route).
 */
async function renderThumb(
  img: DecodedRgb8, x: number, y: number, w: number, h: number,
): Promise<string> {
  // Build an ImageData for the patch (RGBA — canvas requires alpha).
  const patch = new Uint8ClampedArray(w * h * 4)
  for (let row = 0; row < h; row++) {
    const yy = y + row
    let inOff  = (yy * img.width + x) * 3
    let outOff = row * w * 4
    for (let col = 0; col < w; col++) {
      patch[outOff]     = img.data[inOff]
      patch[outOff + 1] = img.data[inOff + 1]
      patch[outOff + 2] = img.data[inOff + 2]
      patch[outOff + 3] = 255
      inOff  += 3
      outOff += 4
    }
  }
  const patchData = new ImageData(patch, w, h)

  const fullCanvas = new OffscreenCanvas(w, h)
  const fullCtx = fullCanvas.getContext('2d')
  if (!fullCtx) throw new Error('OffscreenCanvas 2D context unavailable')
  fullCtx.putImageData(patchData, 0, 0)

  const thumbCanvas = new OffscreenCanvas(SCAN_THUMB_SIZE, SCAN_THUMB_SIZE)
  const thumbCtx = thumbCanvas.getContext('2d')
  if (!thumbCtx) throw new Error('OffscreenCanvas 2D context unavailable')
  thumbCtx.drawImage(fullCanvas, 0, 0, SCAN_THUMB_SIZE, SCAN_THUMB_SIZE)

  const blob = await thumbCanvas.convertToBlob({ type: 'image/png' })
  return arrayBufferToBase64(await blob.arrayBuffer())
}

/**
 * Build a grayscale |src - dst| thumbnail at 1× amplification. Each pixel is
 * the channel-mean absolute difference (matches the metric used for
 * maxDiff / the skip-empty threshold). The dialog applies CSS
 * `filter: brightness(amp)` on render to amplify without re-encoding —
 * scrubbing the amp slider is a pure style change, no worker round-trip.
 */
async function renderDiffThumb(
  sImg: DecodedRgb8, dImg: DecodedRgb8,
  x: number, y: number, w: number, h: number,
): Promise<string> {
  const patch = new Uint8ClampedArray(w * h * 4)
  for (let row = 0; row < h; row++) {
    const yy = y + row
    let sOff   = (yy * sImg.width + x) * 3
    let dOff   = (yy * dImg.width + x) * 3
    let outOff = row * w * 4
    for (let col = 0; col < w; col++) {
      // Channel-mean of |delta| — matches preview-filter-core.patchMaxAbsDiff
      // so the visual map agrees with the threshold metric.
      const dr = Math.abs(sImg.data[sOff]     - dImg.data[dOff])
      const dg = Math.abs(sImg.data[sOff + 1] - dImg.data[dOff + 1])
      const db = Math.abs(sImg.data[sOff + 2] - dImg.data[dOff + 2])
      const g  = (dr + dg + db) / 3
      patch[outOff]     = g
      patch[outOff + 1] = g
      patch[outOff + 2] = g
      patch[outOff + 3] = 255
      sOff   += 3
      dOff   += 3
      outOff += 4
    }
  }
  const patchData = new ImageData(patch, w, h)

  const fullCanvas = new OffscreenCanvas(w, h)
  const fullCtx = fullCanvas.getContext('2d')
  if (!fullCtx) throw new Error('OffscreenCanvas 2D context unavailable')
  fullCtx.putImageData(patchData, 0, 0)

  const thumbCanvas = new OffscreenCanvas(SCAN_THUMB_SIZE, SCAN_THUMB_SIZE)
  const thumbCtx = thumbCanvas.getContext('2d')
  if (!thumbCtx) throw new Error('OffscreenCanvas 2D context unavailable')
  thumbCtx.drawImage(fullCanvas, 0, 0, SCAN_THUMB_SIZE, SCAN_THUMB_SIZE)

  const blob = await thumbCanvas.convertToBlob({ type: 'image/png' })
  return arrayBufferToBase64(await blob.arrayBuffer())
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  // Chunk to avoid blowing the call-stack with huge strings on big PNGs
  // (a 96×96 PNG is well under 64KB so this is conservative).
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
