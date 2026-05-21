/**
 * Pure-JS preview-filter helpers shared between the browser worker and the
 * (legacy) server route. No Node-only or browser-only deps live here — the
 * worker imports parse-exr separately and the server route adds sharp on top.
 *
 * The patch-skip metric must stay byte-identical to tunet.py:302 so the
 * preview matches what training will actually skip:
 *
 *     metric = np.abs(src - dst).mean(axis=2).max()
 *
 * (channel-mean of |delta|, then max over patch pixels)
 */

export const SCAN_THUMB_SIZE = 96
export const MAX_PATCHES     = 200

export interface PatchPlan {
  fileIdx: number   // index into the input pair list
  x:       number
  y:       number
  w:       number
  h:       number
}

/**
 * Compute the channel-mean-of-|delta|, max-over-pixels metric over a patch
 * within a full-image RGB byte buffer. `imgW` is the image width in pixels;
 * `src`/`dst` are tightly packed RGB bytes (3 bytes per pixel).
 */
export function patchMaxAbsDiff(
  src: Uint8Array | Uint8ClampedArray,
  dst: Uint8Array | Uint8ClampedArray,
  imgW: number,
  x: number, y: number, w: number, h: number,
): number {
  let maxDiff = 0
  for (let row = 0; row < h; row++) {
    const yy = y + row
    let off = (yy * imgW + x) * 3
    for (let col = 0; col < w; col++) {
      const dr = Math.abs(src[off]     - dst[off])
      const dg = Math.abs(src[off + 1] - dst[off + 1])
      const db = Math.abs(src[off + 2] - dst[off + 2])
      const meanDiff = (dr + dg + db) / 3
      if (meanDiff > maxDiff) maxDiff = meanDiff
      off += 3
    }
  }
  return maxDiff
}

/**
 * Linear-EXR (Float32 RGBA) → 8-bit gamma-corrected RGB. Approximate sRGB via
 * x^(1/2.2). The skip metric is computed in 8-bit space so this lossy curve
 * is sufficient — true sRGB OETF would be more expensive without changing
 * which patches get filtered.
 *
 * Vertical flip: parse-exr (THREE.EXRLoader lineage) emits scanlines bottom-up
 * because it targets WebGL textures, whose origin is bottom-left — it writes
 * image row `y` to buffer row `height - 1 - y` regardless of the file's
 * lineOrder (parse-exr/index.js:2108). We treat the buffer as a top-down image
 * (row 0 = top), the same convention createImageBitmap gives us for JPEG/PNG,
 * so we flip back here. Without this, EXR previews render upside-down while
 * JPEGs look correct (and EXR↔JPEG pairs would mis-align in the diff metric).
 */
export function exrFloatRgbaToRgb8(src: Float32Array, w: number, h: number): Uint8Array {
  if (src.length !== w * h * 4) {
    throw new Error(`unexpected EXR data length (${src.length} vs ${w * h * 4})`)
  }
  const out = new Uint8Array(w * h * 3)
  const invGamma = 1 / 2.2
  for (let y = 0; y < h; y++) {
    let i = (h - 1 - y) * w * 4   // source row, bottom-up
    let j = y * w * 3             // dest row, top-down
    for (let x = 0; x < w; x++) {
      out[j]     = toByteGamma(src[i],     invGamma)
      out[j + 1] = toByteGamma(src[i + 1], invGamma)
      out[j + 2] = toByteGamma(src[i + 2], invGamma)
      i += 4
      j += 3
    }
  }
  return out
}

function toByteGamma(v: number, invGamma: number): number {
  if (!(v > 0)) return 0
  if (v >= 1)   return 255
  return Math.round(Math.pow(v, invGamma) * 255)
}

/**
 * Plan all candidate patches across pairs given dims, resolution, and overlap.
 * Returns `{ plans, stride }` with `plans` sized exactly to the patch grid.
 */
export function planPatches(
  pairDims: { width: number; height: number }[],
  resolution: number,
  overlap: number,
): { plans: PatchPlan[]; stride: number } {
  const stride = Math.max(1, Math.floor(resolution - resolution * overlap))
  const plans: PatchPlan[] = []
  for (let i = 0; i < pairDims.length; i++) {
    const d = pairDims[i]
    if (!d || d.width < resolution || d.height < resolution) continue

    const xs: number[] = []
    const ys: number[] = []
    for (let x = 0; x + resolution <= d.width; x += stride) xs.push(x)
    if (d.width  > resolution && (d.width  - resolution) % stride !== 0) xs.push(d.width  - resolution)
    for (let y = 0; y + resolution <= d.height; y += stride) ys.push(y)
    if (d.height > resolution && (d.height - resolution) % stride !== 0) ys.push(d.height - resolution)

    for (const y of dedup(ys)) {
      for (const x of dedup(xs)) {
        plans.push({ fileIdx: i, x, y, w: resolution, h: resolution })
      }
    }
  }
  return { plans, stride }
}

/**
 * Sample up to `n` items spread evenly across `arr` (deterministic).
 */
export function sampleEvenly<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr.slice()
  const step = arr.length / n
  const out: T[] = []
  for (let i = 0; i < n; i++) out.push(arr[Math.min(arr.length - 1, Math.floor(i * step))])
  return out
}

function dedup(arr: number[]): number[] {
  return [...new Set(arr)].sort((a, b) => a - b)
}
