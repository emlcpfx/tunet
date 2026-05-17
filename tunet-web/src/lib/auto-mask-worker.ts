/// <reference lib="webworker" />

/**
 * Auto-mask preview worker.
 *
 * Replicates the expensive front half of training/loss.py:refine_auto_mask
 * once per sample, then ships the post-sigmoid mask intensity back to the
 * main thread as a Uint8Array. The main thread re-applies the cheap final
 * step — `mask^gamma` — every time the user moves the gamma slider, so the
 * UI stays live without re-decoding or re-blurring per frame.
 *
 * Pipeline (matches training/loss.py:23 verbatim):
 *
 *   1. mean(|src - dst|, channel)   → grayscale [0,1] diff
 *   2. Gaussian blur, kernel = max(31, 31 * res / 256) | 1  (separable)
 *   3. normalize by per-image max  (skip if max < noise_threshold)
 *   4. sigmoid(20 * (x - 0.15))
 *   5. renormalize to [0,1]
 *   6. (then on main thread)  out = x ^ gamma
 *
 * Wire protocol:
 *
 *   in:  { type: 'compute', pairs: AutoMaskPair[], previewSize: number }
 *   in:  { type: 'cancel' }
 *
 *   out: { type: 'sample', sample: AutoMaskSample, index: number, total: number }
 *   out: { type: 'done', emitted: number }
 *   out: { type: 'error', message: string }
 *
 * The `previewSize` is the side length we resize to before computing the
 * mask — keeps each sample fast (~50ms) regardless of source resolution.
 * 256px is a good default; it's the same scale the trainer uses internally
 * when it picks the kernel size.
 */

import parseExr from 'parse-exr'
import { exrFloatRgbaToRgb8 } from './preview-filter-core'

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope

export interface AutoMaskPair {
  name: string
  src:  File
  dst:  File
}

export interface AutoMaskSample {
  name:   string
  /** RGBA-PNG of the src thumbnail (base64) — what the user looks AT */
  srcPng: string
  /** Pre-gamma mask intensity, one byte per pixel, length = previewSize² */
  mask:   Uint8Array
  size:   number
  /** True if max diff was below noise threshold — entire mask is zero */
  empty:  boolean
}

interface ComputeMsg { type: 'compute'; pairs: AutoMaskPair[]; previewSize: number }
interface CancelMsg  { type: 'cancel' }
type InMsg = ComputeMsg | CancelMsg

interface DecodedRgb8 { data: Uint8Array; width: number; height: number }

const NOISE_THRESHOLD = 0.01   // matches training/loss.py:23 default

let cancelled = false

ctx.addEventListener('message', (e: MessageEvent<InMsg>) => {
  const msg = e.data
  if (msg.type === 'cancel') { cancelled = true; return }
  if (msg.type === 'compute') {
    cancelled = false
    runCompute(msg).catch(err =>
      post({ type: 'error', message: err?.message ?? String(err) }))
  }
})

async function runCompute(msg: ComputeMsg) {
  const { pairs, previewSize } = msg
  if (pairs.length === 0) {
    post({ type: 'error', message: 'no pairs supplied' })
    return
  }

  let emitted = 0
  for (let i = 0; i < pairs.length; i++) {
    if (cancelled) return
    try {
      const sample = await computeSample(pairs[i], previewSize)
      // Transfer the mask buffer to avoid a copy (it's the only large field).
      post({ type: 'sample', sample, index: i, total: pairs.length },
            [sample.mask.buffer])
      emitted++
    } catch (err) {
      // Skip the sample but keep going — partial results are useful.
      console.warn('auto-mask sample failed:', err)
    }
  }
  post({ type: 'done', emitted })
}

async function computeSample(pair: AutoMaskPair, previewSize: number): Promise<AutoMaskSample> {
  // Decode both images at full res, then downsample to previewSize.
  const src = await decode(pair.src)
  const dst = await decode(pair.dst)
  if (src.width !== dst.width || src.height !== dst.height) {
    throw new Error('src/dst size mismatch')
  }

  // Downsample with the canvas — gives us a clean bilinear shrink for free,
  // and we want the same downsampled rgb to make a thumbnail anyway.
  const srcSmall = await resizeRgb(src, previewSize)
  const dstSmall = await resizeRgb(dst, previewSize)

  // ── Step 1: per-channel mean |src - dst|, normalize to [0, 1] ────────────
  const N = previewSize * previewSize
  const diff = new Float32Array(N)
  for (let i = 0, j = 0; i < diff.length; i++, j += 3) {
    const dr = Math.abs(srcSmall.data[j]     - dstSmall.data[j])
    const dg = Math.abs(srcSmall.data[j + 1] - dstSmall.data[j + 1])
    const db = Math.abs(srcSmall.data[j + 2] - dstSmall.data[j + 2])
    diff[i] = (dr + dg + db) / 3 / 255
  }

  // ── Step 2: Gaussian blur ────────────────────────────────────────────────
  // Mirror the kernel size formula from refine_auto_mask. At previewSize=256
  // this collapses to k=31, which is what the trainer would also pick.
  const kernelSize = Math.max(31, Math.floor(31 * previewSize / 256)) | 1
  const blurred = gaussianBlur(diff, previewSize, previewSize, kernelSize)

  // ── Step 3: per-image max + early-out for noise ──────────────────────────
  let maxVal = 0
  for (let i = 0; i < blurred.length; i++) if (blurred[i] > maxVal) maxVal = blurred[i]
  maxVal = maxVal + 1e-8

  const empty = maxVal <= NOISE_THRESHOLD
  if (empty) {
    return {
      name:   pair.name,
      srcPng: await rgbToPng(srcSmall),
      mask:   new Uint8Array(N),
      size:   previewSize,
      empty,
    }
  }

  // ── Step 4: sigmoid(20 * (x/max - 0.15)) ─────────────────────────────────
  const sig = new Float32Array(N)
  let sigMin = Infinity, sigMax = -Infinity
  for (let i = 0; i < N; i++) {
    const v = 1 / (1 + Math.exp(-20 * ((blurred[i] / maxVal) - 0.15)))
    sig[i] = v
    if (v < sigMin) sigMin = v
    if (v > sigMax) sigMax = v
  }

  // ── Step 5: renormalize to [0, 1] ────────────────────────────────────────
  const range = (sigMax - sigMin) + 1e-8
  const mask = new Uint8Array(N)
  for (let i = 0; i < N; i++) {
    const v = (sig[i] - sigMin) / range
    mask[i] = Math.max(0, Math.min(255, Math.round(v * 255)))
  }

  return {
    name:   pair.name,
    srcPng: await rgbToPng(srcSmall),
    mask,
    size:   previewSize,
    empty,
  }
}

// ── Decode helpers (same approach as preview-filter-worker) ─────────────────

async function decode(file: File): Promise<DecodedRgb8> {
  return extName(file.name) === '.exr' ? decodeExr(file) : decodeBitmap(file)
}

async function decodeBitmap(file: File): Promise<DecodedRgb8> {
  const bmp = await createImageBitmap(file)
  const w = bmp.width, h = bmp.height
  const canvas = new OffscreenCanvas(w, h)
  const gctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!gctx) { bmp.close(); throw new Error('OffscreenCanvas 2D context unavailable') }
  gctx.drawImage(bmp, 0, 0)
  bmp.close()
  const img = gctx.getImageData(0, 0, w, h)
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
  const exr = parseExr(ab, 1015)
  return {
    data:   exrFloatRgbaToRgb8(exr.data as Float32Array, exr.width, exr.height),
    width:  exr.width,
    height: exr.height,
  }
}

/** Bilinear shrink of an RGB8 buffer to a target square. */
async function resizeRgb(img: DecodedRgb8, target: number): Promise<DecodedRgb8> {
  const rgba = new Uint8ClampedArray(img.width * img.height * 4)
  for (let i = 0, j = 0; i < img.data.length; i += 3, j += 4) {
    rgba[j]     = img.data[i]
    rgba[j + 1] = img.data[i + 1]
    rgba[j + 2] = img.data[i + 2]
    rgba[j + 3] = 255
  }
  const srcImg = new ImageData(rgba, img.width, img.height)
  const srcCanvas = new OffscreenCanvas(img.width, img.height)
  const srcCtx = srcCanvas.getContext('2d')
  if (!srcCtx) throw new Error('OffscreenCanvas 2D context unavailable')
  srcCtx.putImageData(srcImg, 0, 0)

  const out = new OffscreenCanvas(target, target)
  const outCtx = out.getContext('2d')
  if (!outCtx) throw new Error('OffscreenCanvas 2D context unavailable')
  outCtx.imageSmoothingEnabled = true
  outCtx.imageSmoothingQuality = 'high'
  outCtx.drawImage(srcCanvas, 0, 0, target, target)
  const small = outCtx.getImageData(0, 0, target, target)
  const rgb = new Uint8Array(target * target * 3)
  for (let i = 0, j = 0; i < small.data.length; i += 4, j += 3) {
    rgb[j]     = small.data[i]
    rgb[j + 1] = small.data[i + 1]
    rgb[j + 2] = small.data[i + 2]
  }
  return { data: rgb, width: target, height: target }
}

async function rgbToPng(img: DecodedRgb8): Promise<string> {
  const rgba = new Uint8ClampedArray(img.width * img.height * 4)
  for (let i = 0, j = 0; i < img.data.length; i += 3, j += 4) {
    rgba[j]     = img.data[i]
    rgba[j + 1] = img.data[i + 1]
    rgba[j + 2] = img.data[i + 2]
    rgba[j + 3] = 255
  }
  const data = new ImageData(rgba, img.width, img.height)
  const canvas = new OffscreenCanvas(img.width, img.height)
  const gctx = canvas.getContext('2d')
  if (!gctx) throw new Error('OffscreenCanvas 2D context unavailable')
  gctx.putImageData(data, 0, 0)
  const blob = await canvas.convertToBlob({ type: 'image/png' })
  return arrayBufferToBase64(await blob.arrayBuffer())
}

/**
 * Separable Gaussian blur, sigma chosen so the kernel approximates a
 * symmetric bell. The trainer uses torch's gaussian_blur which auto-derives
 * sigma from the kernel size; we mirror that with sigma = (k - 1) / 6 so
 * ±3σ covers the kernel radius. Reflect-pad at the edges.
 */
function gaussianBlur(src: Float32Array, w: number, h: number, k: number): Float32Array {
  const sigma  = Math.max(1, (k - 1) / 6)
  const radius = (k - 1) >> 1
  const kernel = new Float32Array(k)
  const twoSigma2 = 2 * sigma * sigma
  let sum = 0
  for (let i = 0; i < k; i++) {
    const x = i - radius
    kernel[i] = Math.exp(-(x * x) / twoSigma2)
    sum += kernel[i]
  }
  for (let i = 0; i < k; i++) kernel[i] /= sum

  // Horizontal pass
  const tmp = new Float32Array(src.length)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0
      for (let i = 0; i < k; i++) {
        let xi = x + (i - radius)
        if (xi < 0) xi = -xi
        else if (xi >= w) xi = 2 * w - xi - 2
        if (xi < 0) xi = 0
        if (xi >= w) xi = w - 1
        acc += src[y * w + xi] * kernel[i]
      }
      tmp[y * w + x] = acc
    }
  }
  // Vertical pass
  const out = new Float32Array(src.length)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0
      for (let i = 0; i < k; i++) {
        let yi = y + (i - radius)
        if (yi < 0) yi = -yi
        else if (yi >= h) yi = 2 * h - yi - 2
        if (yi < 0) yi = 0
        if (yi >= h) yi = h - 1
        acc += tmp[yi * w + x] * kernel[i]
      }
      out[y * w + x] = acc
    }
  }
  return out
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

function post(msg: unknown, transfer?: Transferable[]) {
  if (transfer) ctx.postMessage(msg, transfer)
  else ctx.postMessage(msg)
}
