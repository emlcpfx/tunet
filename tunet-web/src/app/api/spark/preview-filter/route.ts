/**
 * POST /api/spark/preview-filter — legacy server-side fallback.
 *
 * The Preview Filter dialog now runs entirely in-browser via a Web Worker
 * (see lib/preview-filter-worker.ts), so this route is no longer the primary
 * codepath. It's kept for parity / future server-staged use cases. All the
 * pure-JS helpers (patch metric, EXR tonemap, plan/sample) live in
 * lib/preview-filter-core.ts and are shared with the worker.
 *
 * Wire format (unchanged):
 *
 *   {
 *     resolution, stride, total, sampled,
 *     patches: [{ x, y, src, maxDiff, thumb }, ...]
 *   }
 */

import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import sharp from 'sharp'
import parseExr from 'parse-exr'
import {
  patchMaxAbsDiff, planPatches, sampleEvenly,
  exrFloatRgbaToRgb8,
  MAX_PATCHES, SCAN_THUMB_SIZE,
} from '@/lib/preview-filter-core'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const maxDuration = 60

// Keep in sync with folder-picker.tsx IMAGE_EXTS. EXR is handled out-of-band
// via parse-exr (sharp's default prebuild can't decode OpenEXR).
const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff', '.exr'])

interface Body {
  stageId?:    string
  resolution?: number   // patch size, default 512
  overlap?:    number   // overlap factor, default 0.25
}

interface PatchOut {
  x:        number
  y:        number
  src:      string
  maxDiff:  number
  thumb:    string   // base64 PNG (data URI body, no prefix)
}

export async function POST(req: Request) {
  let body: Body
  try { body = await req.json() } catch { return jsonError('Invalid JSON', 400) }

  if (!body.stageId) return jsonError('stageId is required', 400)
  const stageRoot = path.join(os.tmpdir(), 'tunet-stages', body.stageId)
  if (!fs.existsSync(stageRoot)) return jsonError('stage not found', 404)

  const srcDir = path.join(stageRoot, 'src')
  const dstDir = path.join(stageRoot, 'dst')
  if (!fs.existsSync(srcDir) || !fs.existsSync(dstDir)) {
    return jsonError('stage missing src/ or dst/', 400)
  }

  const resolution = body.resolution ?? 512
  const overlap    = body.overlap ?? 0.25
  const stride     = Math.max(1, Math.floor(resolution - resolution * overlap))

  // Match basenames between src/ and dst/.
  const pairs = matchPairs(srcDir, dstDir, IMAGE_EXTS)
  if (pairs.length === 0) {
    const srcAll = fs.existsSync(srcDir) ? fs.readdirSync(srcDir) : []
    const dstAll = fs.existsSync(dstDir) ? fs.readdirSync(dstDir) : []
    const srcImg = srcAll.filter(n => IMAGE_EXTS.has(path.extname(n).toLowerCase()))
    const dstImg = dstAll.filter(n => IMAGE_EXTS.has(path.extname(n).toLowerCase()))

    if (srcImg.length === 0 || dstImg.length === 0) {
      return jsonError(
        `no images in ${srcImg.length === 0 ? 'src/' : 'dst/'} ` +
        `(src: ${srcAll.length} files, dst: ${dstAll.length} files)`,
        400,
      )
    }
    const sample = (arr: string[]) => arr.slice(0, 3).join(', ') + (arr.length > 3 ? `, … (${arr.length})` : '')
    return jsonError(
      `src/ and dst/ filenames don't match. src: [${sample(srcImg)}]  dst: [${sample(dstImg)}]`,
      400,
    )
  }

  // Probe dims for every pair (warm decodeCache for EXR so the patch loop
  // doesn't re-decode the same file).
  const decodeCache = new Map<string, DecodedRgb8>()
  const dims: { width: number; height: number }[] = []
  const validPairs: typeof pairs = []
  for (const pair of pairs) {
    try {
      const d = await readDims(pair.srcPath, decodeCache)
      if (d.width < resolution || d.height < resolution) continue
      dims.push(d)
      validPairs.push(pair)
    } catch {
      // skip unreadable file
    }
  }

  if (validPairs.length === 0) return jsonError('no valid patches (images too small?)', 400)

  const { plans, stride: stride2 } = planPatches(dims, resolution, overlap)
  void stride2  // stride from helper matches our local stride
  if (plans.length === 0) return jsonError('no valid patches (images too small?)', 400)

  const sampled = plans.length <= MAX_PATCHES ? plans : sampleEvenly(plans, MAX_PATCHES)

  // Per-patch decode reuses the cache we already warmed during dim probes.
  // EXR goes via parse-exr + tonemap; everything else through sharp.
  const decode = async (p: string): Promise<DecodedRgb8> => {
    const cached = decodeCache.get(p)
    if (cached) return cached
    const out = path.extname(p).toLowerCase() === '.exr'
      ? await decodeExr(p)
      : await decodeSharp(p)
    decodeCache.set(p, out)
    return out
  }

  const out: PatchOut[] = []
  for (const plan of sampled) {
    const pair = validPairs[plan.fileIdx]
    try {
      const sImg = await decode(pair.srcPath)
      const dImg = await decode(pair.dstPath)
      if (sImg.info.width !== dImg.info.width || sImg.info.height !== dImg.info.height) continue
      if (sImg.info.channels !== 3 || dImg.info.channels !== 3) continue

      const maxDiff = patchMaxAbsDiff(
        sImg.data, dImg.data, sImg.info.width,
        plan.x, plan.y, plan.w, plan.h,
      )

      const thumb = await sharp(sImg.data, {
        raw: { width: sImg.info.width, height: sImg.info.height, channels: 3 },
      })
        .extract({ left: plan.x, top: plan.y, width: plan.w, height: plan.h })
        .resize(SCAN_THUMB_SIZE, SCAN_THUMB_SIZE, { fit: 'fill' })
        .png({ compressionLevel: 6 })
        .toBuffer()

      out.push({
        x: plan.x,
        y: plan.y,
        src: path.basename(pair.srcPath),
        maxDiff,
        thumb: thumb.toString('base64'),
      })
    } catch {
      // skip patch on any decode/extract error
    }
  }

  return new Response(JSON.stringify({
    resolution, stride,
    total:  plans.length,
    sampled: out.length,
    patches: out,
  }), { headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function matchPairs(srcDir: string, dstDir: string, exts: Set<string>): { srcPath: string; dstPath: string }[] {
  const srcFiles = listImages(srcDir, exts)
  const dstByBase = new Map<string, string>()
  for (const f of listImages(dstDir, exts)) dstByBase.set(stripExt(path.basename(f)), f)
  const out: { srcPath: string; dstPath: string }[] = []
  for (const s of srcFiles) {
    const m = dstByBase.get(stripExt(path.basename(s)))
    if (m) out.push({ srcPath: s, dstPath: m })
  }
  return out
}

function listImages(dir: string, exts: Set<string>): string[] {
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir)
    .filter(name => exts.has(path.extname(name).toLowerCase()))
    .sort()
    .map(name => path.join(dir, name))
}

function stripExt(name: string): string {
  return name.replace(/\.[^.]+$/, '').toLowerCase()
}

// ── Image decode (sharp + EXR) ──────────────────────────────────────────────

interface DecodedRgb8 { data: Buffer; info: { width: number; height: number; channels: 3 } }

async function readDims(
  p: string,
  cache: Map<string, DecodedRgb8>,
): Promise<{ width: number; height: number }> {
  if (path.extname(p).toLowerCase() === '.exr') {
    // parse-exr decodes the whole file even just for dims, so we cache the
    // result here so the per-patch decode loop doesn't have to do it again.
    const cached = cache.get(p)
    if (cached) return { width: cached.info.width, height: cached.info.height }
    const decoded = await decodeExr(p)
    cache.set(p, decoded)
    return { width: decoded.info.width, height: decoded.info.height }
  }
  const meta = await sharp(p).metadata()
  if (!meta.width || !meta.height) throw new Error('no dims')
  return { width: meta.width, height: meta.height }
}

async function decodeSharp(p: string): Promise<DecodedRgb8> {
  const out = await sharp(p).removeAlpha().raw().toBuffer({ resolveWithObject: true })
  return {
    data: out.data,
    info: { width: out.info.width, height: out.info.height, channels: 3 },
  }
}

/**
 * Decode an OpenEXR file to 8-bit linear-tonemapped RGB. Tonemap math lives
 * in lib/preview-filter-core.ts and is shared with the browser worker.
 */
async function decodeExr(p: string): Promise<DecodedRgb8> {
  const buf  = await fs.promises.readFile(p)
  // parse-exr expects an ArrayBuffer, not a Node Buffer. Slice to the exact
  // backing range so we don't accidentally pass an oversized pool buffer.
  const ab   = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
  // Use Float (1015) — Half-float (1016) returns Uint16Array half-bits which
  // would need extra unpacking on our side.
  const exr  = parseExr(ab, 1015)
  const w = exr.width
  const h = exr.height
  const rgb = exrFloatRgbaToRgb8(exr.data as Float32Array, w, h)
  // Wrap as Buffer so sharp's raw input accepts it without a copy.
  const out = Buffer.from(rgb.buffer, rgb.byteOffset, rgb.byteLength)
  return { data: out, info: { width: w, height: h, channels: 3 } }
}
