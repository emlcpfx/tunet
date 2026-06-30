/**
 * GET /api/spark/jobs/:id/files
 *
 *   ?path=<rel>        — proxy a single file from ShareSync.
 *                         Returns the file body with the upstream Content-Type.
 *                         Add ?download=1 to set Content-Disposition.
 *   (no path)          — list the job's output dir (one level, JSON).
 *
 * The ShareSync bearer token stays server-side. Browsers can't reach Spark's
 * WebDAV directly with our token, so this is the only path.
 *
 * IMPORTANT: paths are scoped to the job's output_share_sync_path. We reject
 * any '..' segments to prevent traversal even though the token couldn't reach
 * outside the job's own dir anyway.
 */

import { getJob, fetchOutputFile, listOutputDir, propfindDir, shareSyncBaseUrl, resolveFilesBase, getToken } from '@/lib/spark'
import type { SparkJob } from '@/lib/spark'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

// Status mapping notes:
//
//   - The job exists but its ShareSync output dir doesn't yet (provisioning,
//     or pre-first-checkpoint). Spark's WebDAV returns 404 on PROPFIND. We
//     turn that into 200 with `{ pending: true, entries: [] }` so the UI
//     can show a "waiting for first upload" state instead of a hard error.
//
//   - A *specific* file is missing (e.g. preview JPG hasn't been written).
//     Spark sometimes returns 404 here, sometimes 500 (observed during job
//     provisioning). We collapse all "looks-empty" upstream codes into a
//     single 404 so the client logic stays simple.
//
//   - If we don't have a ShareSync base URL configured at all, that's a
//     server-config problem — return 503 with a clear message (not 502 which
//     implies an upstream failure).
const PENDING_UPSTREAM_STATUSES = new Set([404, 500, 502, 503])

export async function GET(
  req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const { id } = await ctx.params
  const url    = new URL(req.url)
  const path   = url.searchParams.get('path')
  const download = url.searchParams.get('download') === '1'

  let job
  try {
    job = await getJob(id)
  } catch (e) {
    return jsonError(e instanceof Error ? e.message : 'failed', 502)
  }
  if (!job) return jsonError('job not found', 404)

  // Resolve the authoritative ShareSync base (v1.23 §3.4) before the sync guard,
  // so a fresh worker with an empty discovered-cache still finds it.
  await resolveFilesBase()
  // Without a base URL we can't reach ShareSync at all — surface as a
  // distinct 503 so the UI can suggest the env var.
  if (!shareSyncBaseUrl(job)) {
    return jsonError(
      'ShareSync base URL not configured (set SPARK_FILES_BASE_URL in .env.local)',
      503,
    )
  }

  // ── Debug mode ──────────────────────────────────────────────────────────
  // When the live preview/downloads can't find a file, this dumps:
  //   - the URL we're computing
  //   - what's actually present at output_share_sync_path (one level deep)
  //   - what's present in the per-job subdir (one level deep)
  // so we can see whether the trainer wrote where we think.
  if (url.searchParams.get('debug') === '1') {
    return debugDump(job)
  }

  // ── List mode ────────────────────────────────────────────────────────────
  if (!path) {
    try {
      const entries = await listOutputFiles(job)
      return new Response(JSON.stringify({ entries }), {
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      })
    } catch (e) {
      // Tag pending vs real failure based on upstream status (encoded by
      // listOutputDir as "PROPFIND failed: HTTP 404" etc.).
      const msg = e instanceof Error ? e.message : 'list failed'
      const m = msg.match(/HTTP (\d+)/)
      const upstream = m ? parseInt(m[1], 10) : 0
      if (PENDING_UPSTREAM_STATUSES.has(upstream)) {
        return new Response(JSON.stringify({ entries: [], pending: true, upstream }), {
          headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
        })
      }
      return jsonError(msg, 502)
    }
  }

  // ── Fetch mode ───────────────────────────────────────────────────────────
  if (path.includes('..') || path.startsWith('/')) {
    return jsonError('invalid path', 400)
  }

  let upstream: Response
  try {
    upstream = await fetchOutputFile(job, path)
  } catch (e) {
    return jsonError(e instanceof Error ? e.message : 'fetch failed', 502)
  }
  if (!upstream.ok) {
    // Collapse "missing file" responses into a clean 404 so the preview
    // component's existing 404 handling kicks in.
    const status = PENDING_UPSTREAM_STATUSES.has(upstream.status) ? 404 : upstream.status
    return jsonError(`upstream HTTP ${upstream.status}`, status)
  }

  const headers = new Headers()
  const ct = upstream.headers.get('Content-Type')
  if (ct) headers.set('Content-Type', ct)
  const cl = upstream.headers.get('Content-Length')
  if (cl) headers.set('Content-Length', cl)
  const lm = upstream.headers.get('Last-Modified')
  if (lm) headers.set('Last-Modified', lm)
  // Allow short browser caching for preview JPGs while a job is running. The
  // client cache-busts via ?v=<step> when it wants a fresh frame.
  headers.set('Cache-Control', 'private, max-age=2')

  if (download) {
    const fname = path.split('/').pop() ?? 'download'
    headers.set('Content-Disposition', `attachment; filename="${fname}"`)
  }

  return new Response(upstream.body, { status: 200, headers })
}

function jsonError(msg: string, status: number): Response {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

interface ListedFile { name: string; size: number; modified: string | null }

/**
 * List a job's downloadable files: the top-level output dir PLUS a one-level
 * descent into `exports/<sub>/` (flame, nuke, …). The auto-export + the
 * "Export now" job write their ONNX / Nuke deliverables two levels deep
 * (train.py → exporters/auto_export.py: `exports/flame/*.onnx`,
 * `exports/nuke/*.{pt,nk,cat}`), but a Depth-1 PROPFIND only sees the
 * `exports/` directory itself — so without this descent the Downloads panel's
 * Flame/Nuke cards never find their files and stay greyed out.
 *
 * Nested files carry their relative path as `name` (e.g.
 * "exports/flame/foo.onnx") so the fetch route can resolve them and the panel
 * can classify by subfolder. We descend only into `exports/` (not arbitrary
 * dirs) to keep this bounded to ~1 + N extra PROPFINDs per list call.
 */
async function listOutputFiles(job: SparkJob): Promise<ListedFile[]> {
  const base = shareSyncBaseUrl(job)
  if (!base) throw new Error('No ShareSync URL available for this job')

  const top = await listOutputDir(job)
  const files: ListedFile[] = top
    .filter(e => !e.isDir)
    .map(e => ({ name: e.name, size: e.size, modified: e.modified }))

  const exportsDir = top.find(e => e.isDir && e.name.toLowerCase() === 'exports')
  if (exportsDir) {
    const subs = await propfindDir(`${base}/exports`)
    for (const sub of subs) {
      if (!sub.isDir) continue
      const subFiles = await propfindDir(`${base}/exports/${encodeURIComponent(sub.name)}`)
      for (const f of subFiles) {
        if (f.isDir) continue
        files.push({ name: `exports/${sub.name}/${f.name}`, size: f.size, modified: f.modified })
      }
    }
  }
  return files
}

/**
 * Debug helper: shows what we *think* the WebDAV URL is plus what's actually
 * sitting at the parent path and at the per-job subdir. Use when the live
 * preview/downloads can't resolve files.
 *
 * Hit `/api/spark/jobs/<id>/files?debug=1` and look at:
 *   - shareSyncBaseUrl       : what we'd PROPFIND for the listing
 *   - parent.entries         : what's directly under output_share_sync_path
 *   - subdir.entries         : what's under <output>/<jobname-subdir>/
 *
 * If `subdir.entries` is empty but `parent.entries` shows training_preview.jpg,
 * the per-job subdir guess is wrong and we should drop it.
 */
async function debugDump(job: import('@/lib/spark').SparkJob): Promise<Response> {
  const out: Record<string, unknown> = {
    job: {
      id:                       job.id,
      status:                   job.status,
      command:                  job.command,
      output_share_sync_path:   job.output_share_sync_path,
      env: {
        TUNET_JOB_NAME:  job.env?.TUNET_JOB_NAME,
        TUNET_FILES_BASE: job.env?.TUNET_FILES_BASE ? '<set>' : undefined,
      },
    },
    computed: {
      shareSyncBaseUrl_with_subdir: shareSyncBaseUrl(job),
    },
  }

  // Probe parent + subdir
  const filesBase = job.env?.TUNET_FILES_BASE ?? process.env.SPARK_FILES_BASE_URL
  const path      = job.output_share_sync_path
  if (!filesBase || !path) {
    out.error = 'missing filesBase or output_share_sync_path'
    return new Response(JSON.stringify(out, null, 2), {
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const parentUrl = `${filesBase.replace(/\/+$/, '')}/${path.replace(/^\/+/, '').replace(/\/+$/, '')}`
  out.computed_parent_url = parentUrl

  try {
    out.parent = await propfindList(parentUrl)
  } catch (e) {
    out.parent = { error: e instanceof Error ? e.message : String(e) }
  }

  // Try the subdir we'd append
  const subdir = (() => {
    const cmd = job.command
    if (!Array.isArray(cmd) || cmd.length < 3) return null
    const a = cmd[2]
    if (typeof a !== 'string' || !a.startsWith('/output/')) return null
    return a.slice('/output/'.length).split('/').filter(Boolean)[0] ?? null
  })()
  out.computed_subdir = subdir

  if (subdir) {
    const subUrl = `${parentUrl}/${encodeURIComponent(subdir)}`
    out.computed_subdir_url = subUrl
    try {
      out.subdir = await propfindList(subUrl)
    } catch (e) {
      out.subdir = { error: e instanceof Error ? e.message : String(e) }
    }
  }

  // ── Climb the URL one segment at a time and PROPFIND each level. The
  // first level that returns a real listing (status 207, non-empty) is the
  // correct space root. We can use that to derive the right
  // SPARK_FILES_BASE_URL.
  out.path_walk = await walkUp(parentUrl)

  return new Response(JSON.stringify(out, null, 2), {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  })
}

/**
 * Walk up the URL one path segment at a time, PROPFIND'ing each level.
 * Helps diagnose "wrong base URL" issues — the deepest level that returns
 * 207 with entries is the correct space root.
 */
async function walkUp(url: string): Promise<Array<{ url: string; status: number; entryCount: number }>> {
  const u = new URL(url)
  const segs = u.pathname.split('/').filter(Boolean)
  const out: Array<{ url: string; status: number; entryCount: number }> = []
  // Try from the leaf up to (but not including) the root.
  for (let i = segs.length; i > 0; i--) {
    const candidate = `${u.protocol}//${u.host}/${segs.slice(0, i).join('/')}`
    try {
      const r = await propfindList(candidate)
      out.push({ url: candidate, status: r.status, entryCount: r.entries.length })
      // Once we hit a working PROPFIND we keep going up (so the user can see
      // the structure) but we don't probe forever.
    } catch (e) {
      out.push({ url: candidate, status: 0, entryCount: 0 })
    }
    if (out.length >= 6) break
  }
  return out
}

interface DebugEntry { name: string; size: number; isDir: boolean }

async function propfindList(url: string): Promise<{ status: number; entries: DebugEntry[] }> {
  const tok = await getToken()
  const res = await fetch(`${url}/`, {
    method:  'PROPFIND',
    headers: {
      'Authorization': `Bearer ${tok}`,
      'Depth':         '1',
      'Accept':        'application/xml,text/xml',
      'Content-Type':  'application/xml',
    },
    body: `<?xml version="1.0" encoding="utf-8"?>
<propfind xmlns="DAV:"><prop><displayname/><getcontentlength/><resourcetype/></prop></propfind>`,
    cache: 'no-store',
  })
  if (!res.ok) {
    return { status: res.status, entries: [] }
  }
  const xml = await res.text()
  const entries: DebugEntry[] = []
  const responseRe = /<(?:\w+:)?response\b[^>]*>([\s\S]*?)<\/(?:\w+:)?response>/gi
  const get = (block: string, tag: string): string | null => {
    const m = block.match(new RegExp(`<(?:\\w+:)?${tag}\\b[^>]*>([\\s\\S]*?)<\\/(?:\\w+:)?${tag}>`, 'i'))
    return m ? m[1].trim() : null
  }
  let parentPath = ''
  try { parentPath = new URL(url).pathname.replace(/\/+$/, '') } catch { /* ignore */ }

  let m: RegExpExecArray | null
  while ((m = responseRe.exec(xml)) !== null) {
    const block = m[1]
    const hrefRaw = get(block, 'href')
    if (!hrefRaw) continue
    let pathOnly = hrefRaw
    try { pathOnly = new URL(hrefRaw, url).pathname } catch { /* relative */ }
    if (pathOnly.replace(/\/+$/, '') === parentPath) continue
    const isDir = /<(?:\w+:)?collection\b/i.test(get(block, 'resourcetype') ?? '')
    const name = decodeURIComponent(pathOnly.split('/').filter(Boolean).pop() ?? '')
    if (!name) continue
    const size = parseInt(get(block, 'getcontentlength') ?? '0', 10) || 0
    entries.push({ name, size, isDir })
  }
  return { status: res.status, entries }
}
