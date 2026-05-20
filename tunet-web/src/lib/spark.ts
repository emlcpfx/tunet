/**
 * Spark Compute v1 client — server-side only.
 *
 * Endpoints (verified live against api.prod.aapse1.sparkcloud.studio):
 *   POST   /auth/login                              email/password → JWT
 *   GET    /api/compute/skus                        eligible instance types
 *   GET    /api/compute/jobs                        list jobs
 *   POST   /api/compute/jobs                        submit job (auto-prepare)
 *   GET    /api/compute/jobs/:id                    job detail
 *   POST   /api/compute/jobs/:id/cancel             SIGTERM
 *   GET    /api/compute/jobs/:id/logs/stream        SSE log stream
 *
 * Submit response shape (camelCase):
 *   { jobId, status, input: { uploadUrl, uploadMethod }, output: { shareSyncBaseUrl, ... } }
 *
 * Detail/list response shape (snake_case):
 *   { id, status, instance_type_name, image, started_running_at, terminal_at, exit_code, ... }
 *
 * IMPORTANT: This module is server-only. The JWT is cached per-process and
 * never reaches the browser. All client UI calls our /api/spark/* routes.
 */

import 'server-only'

export const SPARK_API = 'https://api.prod.aapse1.sparkcloud.studio'

// Default image — matches what spark_launch.py uses (Walt's last-known-good)
export const DEFAULT_IMAGE = 'runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204'

// ── GPU shortcut → SKU map (mirrors spark_launch.py GPU_TYPES) ────────────────

export const GPU_TYPES = {
  t4:           { sku: 'g4dn.xlarge',   gpu: 'NVIDIA T4',           vramGb: 16, gpuCount: 1, label: 'T4 16GB · slow' },
  a10:          { sku: 'g5.xlarge',     gpu: 'NVIDIA A10',          vramGb: 24, gpuCount: 1, label: 'A10 24GB · cheap' },
  l4:           { sku: 'g6.2xlarge',    gpu: 'NVIDIA L4',           vramGb: 24, gpuCount: 1, label: 'L4 24GB' },
  // 2026-05-01: g6e.4xlarge dropped from Spark allow-list. g6e.8xlarge has
  // the same single L40S — just more host CPU/RAM, similar hourly rate.
  l40s:         { sku: 'g6e.8xlarge',   gpu: 'NVIDIA L40S',         vramGb: 48, gpuCount: 1, label: 'L40S 48GB · recommended' },
  a10x4:        { sku: 'g5.24xlarge',   gpu: 'NVIDIA A10',          vramGb: 24, gpuCount: 4, label: '4× A10 24GB · multi-GPU' },
  l40sx4:       { sku: 'g6e.12xlarge',  gpu: 'NVIDIA L40S',         vramGb: 48, gpuCount: 4, label: '4× L40S 48GB · multi-GPU' },
  // 2026-05-01: g7e.xlarge dropped from allow-list. g7e.2xlarge is the
  // smallest single-GPU RTX PRO 6000 still eligible.
  rtxpro6000:   { sku: 'g7e.2xlarge',   gpu: 'NVIDIA RTX PRO 6000', vramGb: 96, gpuCount: 1, label: 'RTX PRO 6000 96GB · fastest' },
  rtxpro6000x8: { sku: 'g7e.48xlarge',  gpu: 'NVIDIA RTX PRO 6000', vramGb: 96, gpuCount: 8, label: '8× RTX PRO 6000 · ludicrous' },
} as const

export type GpuKey = keyof typeof GPU_TYPES

// ── Auth — lazy JWT cached per server process ────────────────────────────────

let _token: string | null = null
let _expiresAt = 0

function creds() {
  const email    = process.env.SPARK_EMAIL
  const password = process.env.SPARK_PASSWORD
  if (!email || !password) {
    throw new Error('SPARK_EMAIL / SPARK_PASSWORD not configured (.env.local)')
  }
  return { email, password }
}

function jwtExpiry(token: string): number {
  try {
    const payload = JSON.parse(
      Buffer.from(token.split('.')[1], 'base64url').toString(),
    )
    return (payload.exp ?? 0) * 1000
  } catch {
    return 0
  }
}

export async function getToken(): Promise<string> {
  if (_token && _expiresAt > Date.now() + 60_000) return _token
  const { email, password } = creds()
  const res = await fetch(`${SPARK_API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
    cache: 'no-store',
  })
  if (!res.ok) {
    throw new Error(`Spark auth failed: HTTP ${res.status}`)
  }
  const data = await res.json()
  const tok = (data.token ?? data.access_token) as string | undefined
  if (!tok) throw new Error('Spark auth: no token in response')
  _token = tok
  _expiresAt = jwtExpiry(tok)
  return tok
}

async function sparkFetch<T = unknown>(
  method: 'GET' | 'POST' | 'DELETE',
  path: string,
  body?: unknown,
): Promise<T> {
  const tok = await getToken()
  const res = await fetch(`${SPARK_API}${path}`, {
    method,
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${tok}`,
      'Accept':        '*/*',
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    cache: 'no-store',
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Spark ${method} ${path} → HTTP ${res.status}: ${text.slice(0, 200)}`)
  }
  return res.json() as Promise<T>
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface SparkSku {
  instanceType: string
  gpuCount:     number
  gpuType:      string
  gpuMemoryGb:  number
}

/** Snake-case shape returned by GET /api/compute/jobs and /api/compute/jobs/:id */
export interface SparkJob {
  id:                          string
  organisation_id?:            number
  user_id?:                    number
  mode?:                       string
  image?:                      string
  command?:                    string[]
  env?:                        Record<string, string>
  output_share_sync_path?:     string
  instance_type_sku_id?:       number
  instance_type_name?:         string
  status:                      'queued' | 'provisioning' | 'running' | 'succeeded' | 'completed' | 'failed' | 'cancelled' | string
  error_code?:                 string | null
  error_message?:              string | null
  exit_code?:                  number | null
  created_at?:                 string
  started_provisioning_at?:    string | null
  started_running_at?:         string | null
  terminal_at?:                string | null
  cancel_requested_at?:        string | null
  last_agent_heartbeat_at?:    string | null
  cuda_version?:               string
  driver_version?:             string
  gpu_name?:                   string
  log_archive_share_sync_path?: string
  log_archive_uploaded_at?:    string | null
  idle_hold_seconds?:          number
  input_share_sync_path?:      string
  // Spark API doesn't echo `name` back; UI falls back to the id prefix.
  name?:                       string
}

export interface SubmitJobResponse {
  jobId:                string
  status:               string
  imageDigest?:         string
  outputShareSyncPath?: string
  createdAt?:           string
  output: {
    shareSyncPath:    string
    shareSyncBaseUrl: string
  }
  input: {
    shareSyncPath:    string
    uploadUrl:        string
    uploadMethod:     'PUT'
    exampleCurl:      string
  }
}

export interface SubmitJobInput {
  name:             string
  instanceType:     string                  // raw SKU, e.g. 'g6e.4xlarge'
  image?:           string                  // defaults to DEFAULT_IMAGE
  command:          string[]                // argv inside the container
  idleHoldSeconds?: number                  // 0 = stop immediately on exit
  env?:             Record<string, string>
  /**
   * Compute mode. Defaults to 'instant' (warm-pool, guaranteed availability).
   * 'smart' = preemptible spare capacity, ~60% cheaper but may be reclaimed
   * mid-run (the platform re-queues onto fresh smart capacity up to the retry
   * budget). See Spark Fuse API docs §13.
   */
  mode?: 'instant' | 'smart'
  /**
   * (smart-mode only) How many times the platform may re-launch this job on
   * fresh smart-mode compute if the current attempt is preempted. Default 1,
   * range [0, 5]. Ignored when mode='instant'.
   */
  maxRetriesOnInterrupt?: number
  /**
   * Override the ShareSync path the agent uploads /output/ to. Defaults to
   * `/Spark Fuse Jobs/{jobId}/`. Used by the export-onnx route to point the
   * export job's output at the *source* job's dir so the .onnx files land
   * in the same Downloads panel as the .pth they were built from.
   */
  outputShareSyncPath?: string
}

// ── Public API ───────────────────────────────────────────────────────────────

export async function listSkus(): Promise<SparkSku[]> {
  const res = await sparkFetch<{ skus: SparkSku[] }>('GET', '/api/compute/skus')
  return res.skus ?? []
}

export async function listJobs(): Promise<SparkJob[]> {
  const res = await sparkFetch<{ jobs: SparkJob[] }>('GET', '/api/compute/jobs')
  return res.jobs ?? []
}

export async function getJob(id: string): Promise<SparkJob | null> {
  try {
    return await sparkFetch<SparkJob>('GET', `/api/compute/jobs/${id}`)
  } catch (e) {
    if (e instanceof Error && e.message.includes('HTTP 404')) return null
    throw e
  }
}

export async function cancelJob(id: string): Promise<SparkJob> {
  return sparkFetch<SparkJob>('POST', `/api/compute/jobs/${id}/cancel`)
}

export interface SparkEstimateResponse {
  instanceType:       string
  mode:               'instant' | 'smart'
  rate: {
    billedPerSecondCents:  string
    billedPerHourUsd:      string
  }
  estimate?: {
    billableSeconds: number
    totalCents:      string
    totalUsd:        string
  }
  notes?: string[]
}

export async function estimateJobCost(
  instanceType: string,
  mode: 'instant' | 'smart' = 'instant',
): Promise<SparkEstimateResponse> {
  return sparkFetch<SparkEstimateResponse>('POST', '/api/compute/jobs/estimate', {
    instanceType,
    mode,
  })
}

export async function submitJob(input: SubmitJobInput): Promise<SubmitJobResponse> {
  const mode = input.mode ?? 'instant'
  return sparkFetch<SubmitJobResponse>('POST', '/api/compute/jobs', {
    name:            input.name,
    instanceType:    input.instanceType,
    image:           input.image ?? DEFAULT_IMAGE,
    command:         input.command,
    inputPushMode:   'auto-prepare',
    // Idle-hold is InstantCompute-only (per docs §11.3); Spark ignores it on
    // smart-mode jobs, but omit it anyway so the request body is honest.
    ...(mode === 'instant' ? { idleHoldSeconds: input.idleHoldSeconds ?? 0 } : {}),
    env:             input.env ?? {},
    mode,
    ...(mode === 'smart' && input.maxRetriesOnInterrupt !== undefined
      ? { maxRetriesOnInterrupt: input.maxRetriesOnInterrupt }
      : {}),
    ...(input.outputShareSyncPath
      ? { outputShareSyncPath: input.outputShareSyncPath }
      : {}),
  })
}

/**
 * Upload a packed tarball to the uploadUrl returned by submitJob.
 * Single-sign-on: the same Bearer token works on both api.* and files.*
 */
export async function uploadInputTarball(
  uploadUrl: string,
  body: Buffer | Blob | ArrayBuffer,
): Promise<void> {
  const tok = await getToken()
  const res = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${tok}`,
      'Content-Type':  'application/octet-stream',
    },
    // @ts-expect-error — Buffer is BodyInit-compatible in Node runtime
    body,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Upload failed HTTP ${res.status}: ${text.slice(0, 200)}`)
  }
}

/**
 * Open the SSE log stream and return the upstream Response so the route
 * handler can pipe it directly to the browser. The Bearer token stays
 * server-side; the client just sees an EventSource on /api/spark/jobs/:id/logs.
 */
export async function openLogStream(jobId: string): Promise<Response> {
  const tok = await getToken()
  return fetch(`${SPARK_API}/api/compute/jobs/${jobId}/logs/stream`, {
    headers: {
      'Authorization': `Bearer ${tok}`,
      'Accept':        'text/event-stream',
    },
    cache: 'no-store',
  })
}

// ── ShareSync (WebDAV) — read-only file access for outputs ───────────────────

/**
 * Resolve the WebDAV base URL for a job's output directory.
 *
 * Spark's submit response gives us a fully-qualified `shareSyncBaseUrl`, but
 * the list/detail responses only echo `output_share_sync_path` (the path
 * portion). To reconstruct the full URL we read either:
 *
 *   1. `env.TUNET_FILES_BASE` if we stashed it at submit time (jobs
 *      submitted through this UI), OR
 *   2. `SPARK_FILES_BASE_URL` env on the server (covers all jobs in the
 *      account — Spark's WebDAV root is per-user, so it's a constant)
 *
 * Returns null if neither is available + the path is missing.
 */
/**
 * Percent-encode each segment of a URL path, preserving '/'. Needed because
 * `output_share_sync_path` is a *raw* path string (e.g. `Spark Fuse Jobs/...`
 * with literal spaces). Node's `fetch` tolerates raw spaces (it auto-encodes
 * to `%20`), but strict consumers like the export job's `curl` reject them
 * with "URL using bad/illegal format". So we encode the path here at the one
 * place the full URL is assembled.
 *
 * Idempotent: a segment that's already encoded (`Compute%20Jobs`) is decoded
 * first so we don't double-encode it to `Compute%2520Jobs`. A stray literal
 * '%' that can't be decoded falls back to encoding the segment as-is.
 */
function encodePathSegments(p: string): string {
  return p
    .split('/')
    .map(seg => {
      if (!seg) return seg
      let raw = seg
      try { raw = decodeURIComponent(seg) } catch { /* keep literal */ }
      return encodeURIComponent(raw)
    })
    .join('/')
}

export function shareSyncBaseUrl(j: SparkJob): string | null {
  const path = j.output_share_sync_path
  if (!path) return null
  // Order matters:
  //   1. Discovered cache — written from a real Spark submit response, so
  //      always correct. Wins over per-job env in case the job was submitted
  //      with a stale/wrong SPARK_FILES_BASE_URL value.
  //   2. Per-job env stash — only useful when the cache hasn't been seeded.
  //   3. Process env — the static fallback the user configures by hand.
  const filesBase = readDiscoveredFilesBase()
                 ?? j.env?.TUNET_FILES_BASE
                 ?? process.env.SPARK_FILES_BASE_URL
  if (!filesBase) return null
  // The trainer writes to `/output/<jobname>/...` (we set output_dir to
  // /output/<safeName> at submit time, see api/spark/training-jobs/route.ts).
  // Spark's agent streams the entire /output/ mount to ShareSync at
  // `output_share_sync_path` — so the actual file lives at:
  //   <files-base> / <output_share_sync_path> / <jobname-subdir> / <relpath>
  //
  // Without the subdir we 404 on every file. Recover the subdir name from the
  // command we issued: it's the basename of command[2] when it starts with
  // /output/. Falls back to no subdir if we can't recover (older jobs, or
  // ones submitted from elsewhere with a different layout).
  const subdir = jobOutputSubdir(j)
  const segs = [
    // filesBase is already URL-encoded (it came from a real submit URL, e.g.
    // the space id arrives as `%24`), so we leave it untouched. Only the raw
    // path/subdir need encoding.
    filesBase.replace(/\/+$/, ''),
    encodePathSegments(path.replace(/^\/+/, '').replace(/\/+$/, '')),
    ...(subdir ? [encodePathSegments(subdir)] : []),
  ].filter(s => s.length > 0)
  return segs.join('/')
}

/**
 * Recover the per-job subdirectory name the trainer wrote into. We need this
 * because Spark uploads /output/ wholesale, so the files end up nested under
 * the basename of the path we passed as the trainer's output_dir.
 */
function jobOutputSubdir(j: SparkJob): string | null {
  const cmd = j.command
  if (!Array.isArray(cmd) || cmd.length < 3) return null
  const arg = cmd[2]
  if (typeof arg !== 'string' || !arg.startsWith('/output/')) return null
  const base = arg.slice('/output/'.length).split('/').filter(Boolean)[0]
  return base ?? null
}

/** Same as shareSyncBaseUrl but for the input dir, used for Preview Filter scans. */
export function shareSyncInputBaseUrl(j: SparkJob): string | null {
  const path = j.input_share_sync_path
  if (!path) return null
  const filesBase = readDiscoveredFilesBase()
                 ?? j.env?.TUNET_FILES_BASE
                 ?? process.env.SPARK_FILES_BASE_URL
  if (!filesBase) return null
  return `${filesBase.replace(/\/+$/, '')}/${encodePathSegments(path.replace(/^\/+/, '').replace(/\/+$/, ''))}`
}

/**
 * Discovered ShareSync base URL — written to a tiny on-disk file by the
 * submit route after a successful POST /api/compute/jobs (the response
 * includes a fully-qualified `output.shareSyncBaseUrl`). The proxy reads
 * this on every request as a fallback for jobs whose env doesn't yet
 * carry TUNET_FILES_BASE (e.g. the very first job, or jobs submitted
 * outside this UI).
 *
 * Lives in os.tmpdir() — survives across requests within a process but
 * resets when Vercel/Next gives us a fresh container, which is fine: any
 * subsequent successful submit re-populates it.
 */
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

const DISCOVERED_BASE_FILE = path.join(os.tmpdir(), 'tunet-spark-files-base.txt')

export function readDiscoveredFilesBase(): string | null {
  try {
    if (!fs.existsSync(DISCOVERED_BASE_FILE)) return null
    const v = fs.readFileSync(DISCOVERED_BASE_FILE, 'utf-8').trim()
    return v.length > 0 ? v : null
  } catch {
    return null
  }
}

/**
 * Cache the ShareSync base URL discovered from a submit response.
 * Strips any per-job suffix so the result is reusable across jobs:
 *   submit gives us  https://eric.files.../dav/spaces/<space>/Compute Jobs/<jobid>
 *   we want to cache https://eric.files.../dav/spaces/<space>
 * The job's output_share_sync_path then provides the rest.
 */
export function writeDiscoveredFilesBase(submitBaseUrl: string, jobOutputPath: string | undefined): void {
  if (!submitBaseUrl) return
  let stripped = submitBaseUrl.replace(/\/+$/, '')

  // The submit URL ends with the job's output_share_sync_path, but they
  // mismatch on URL encoding: the URL has `Compute%20Jobs` while the
  // path string has `Compute Jobs`. Decode both sides before comparing.
  if (jobOutputPath) {
    const suffix = jobOutputPath.replace(/^\/+/, '').replace(/\/+$/, '')
    if (suffix) {
      const strippedDecoded = safeDecode(stripped)
      const cmpSuffix       = '/' + suffix
      if (strippedDecoded.endsWith(cmpSuffix)) {
        // Find the corresponding cut point in the original encoded URL by
        // walking back the same number of decoded characters.
        const cutLenDecoded = cmpSuffix.length
        // Decoded → encoded length isn't 1:1 (`%20` is 3 chars vs ` ` is 1),
        // so re-derive by encoding the decoded prefix.
        const prefix = strippedDecoded.slice(0, strippedDecoded.length - cutLenDecoded)
        // Rebuild encoded prefix: split on '/' and encode each segment.
        try {
          const u = new URL(prefix)
          const segs = u.pathname.split('/').filter(Boolean).map(encodeURIComponent)
          stripped = `${u.protocol}//${u.host}/${segs.join('/')}`.replace(/\/+$/, '')
        } catch {
          // Fall back to the decoded prefix as-is if we can't parse it.
          stripped = prefix.replace(/\/+$/, '')
        }
      }
    }
  }

  try {
    fs.writeFileSync(DISCOVERED_BASE_FILE, stripped, 'utf-8')
  } catch {
    /* best-effort */
  }
}

function safeDecode(s: string): string {
  try { return decodeURI(s) } catch { return s }
}

/**
 * Fetch a file from a job's output ShareSync directory.
 * Path is relative to output_share_sync_path, e.g. 'training_preview.jpg'.
 */
export async function fetchOutputFile(
  j: SparkJob,
  relPath: string,
  init?: { method?: 'GET' | 'HEAD'; headers?: Record<string, string> },
): Promise<Response> {
  const base = shareSyncBaseUrl(j)
  if (!base) {
    throw new Error('No ShareSync URL available for this job')
  }
  const safe = relPath.replace(/^\/+/, '').split('/').map(encodeURIComponent).join('/')
  const tok  = await getToken()
  return fetch(`${base}/${safe}`, {
    method: init?.method ?? 'GET',
    headers: {
      'Authorization': `Bearer ${tok}`,
      ...(init?.headers ?? {}),
    },
    cache: 'no-store',
  })
}

interface WebdavEntry {
  name:     string         // basename
  href:     string         // raw href from PROPFIND (server URL-encoded)
  size:     number
  modified: string | null
  isDir:    boolean
}

/**
 * List the contents of a job's output dir via WebDAV PROPFIND (Depth: 1).
 * Returns just the immediate children (no recursion).
 */
export async function listOutputDir(j: SparkJob): Promise<WebdavEntry[]> {
  const base = shareSyncBaseUrl(j)
  if (!base) throw new Error('No ShareSync URL available for this job')
  return propfindDir(base)
}

/**
 * Depth-1 PROPFIND on an arbitrary ShareSync URL. Returns the immediate
 * children. Used by callers that need to look at a sub-directory of a job
 * (e.g. `<base>/exports/flame/`) without enumerating the whole tree.
 *
 * Returns an empty array on 404 — callers usually want "no files" rather
 * than an exception when the sub-directory simply doesn't exist yet.
 */
export async function propfindDir(url: string): Promise<WebdavEntry[]> {
  const tok = await getToken()
  // Ensure trailing slash so the WebDAV server treats it as a collection.
  const dirUrl = url.endsWith('/') ? url : `${url}/`
  const res = await fetch(dirUrl, {
    method: 'PROPFIND',
    headers: {
      'Authorization': `Bearer ${tok}`,
      'Depth':         '1',
      'Accept':        'application/xml,text/xml',
      'Content-Type':  'application/xml',
    },
    body: `<?xml version="1.0" encoding="utf-8"?>
<propfind xmlns="DAV:">
  <prop>
    <displayname/>
    <getcontentlength/>
    <getlastmodified/>
    <resourcetype/>
  </prop>
</propfind>`,
    cache: 'no-store',
  })
  if (res.status === 404) return []
  if (!res.ok) {
    throw new Error(`PROPFIND failed: HTTP ${res.status}`)
  }
  const xml = await res.text()
  return parseWebdavMultistatus(xml, dirUrl)
}

/**
 * Minimal WebDAV multistatus parser. We don't pull in a dependency for this —
 * the response is small and the schema is fixed. Skips the entry whose href
 * matches the parent dir.
 */
function parseWebdavMultistatus(xml: string, parentBase: string): WebdavEntry[] {
  const entries: WebdavEntry[] = []
  // Match <response>...</response> blocks (case-insensitive, possibly namespaced)
  const responseRe = /<(?:\w+:)?response\b[^>]*>([\s\S]*?)<\/(?:\w+:)?response>/gi
  const get = (block: string, tag: string): string | null => {
    const m = block.match(new RegExp(`<(?:\\w+:)?${tag}\\b[^>]*>([\\s\\S]*?)<\\/(?:\\w+:)?${tag}>`, 'i'))
    return m ? m[1].trim() : null
  }

  // Compute the parent's path-only portion to detect self-referential entries
  let parentPath = ''
  try { parentPath = new URL(parentBase).pathname.replace(/\/+$/, '') } catch { /* ignore */ }

  let m: RegExpExecArray | null
  while ((m = responseRe.exec(xml)) !== null) {
    const block = m[1]
    const hrefRaw = get(block, 'href')
    if (!hrefRaw) continue

    // href is URL-encoded; we keep it that way for fetching but decode for the basename
    const hrefPath = hrefRaw.replace(/&amp;/g, '&').replace(/<!\[CDATA\[(.+?)\]\]>/g, '$1').trim()

    // Skip the self entry (the parent dir itself)
    let pathOnly = hrefPath
    try { pathOnly = new URL(hrefPath, parentBase).pathname } catch { /* relative */ }
    const normSelf = pathOnly.replace(/\/+$/, '')
    if (normSelf === parentPath) continue

    const isDir = /<(?:\w+:)?collection\b/i.test(get(block, 'resourcetype') ?? '')
    let name = decodeURIComponent(pathOnly.split('/').filter(Boolean).pop() ?? '')
    if (!name) continue

    const sizeStr = get(block, 'getcontentlength')
    const size    = sizeStr ? parseInt(sizeStr, 10) : 0
    const modified = get(block, 'getlastmodified')

    entries.push({
      name,
      href: hrefPath,
      size: Number.isFinite(size) ? size : 0,
      modified,
      isDir,
    })
  }
  return entries
}

// ── Status helpers (UI shared) ───────────────────────────────────────────────

export const ACTIVE_STATUSES = new Set(['queued', 'provisioning', 'running'])

export function isActive(j: SparkJob): boolean {
  return ACTIVE_STATUSES.has(j.status)
}

export function jobLabel(j: SparkJob): string {
  if (j.name) return j.name
  return j.id.length > 8 ? j.id.slice(0, 8) : j.id
}

export function jobRuntimeMs(j: SparkJob): number | null {
  const start = j.started_running_at ?? j.started_provisioning_at ?? j.created_at
  if (!start) return null
  const startMs = Date.parse(start)
  const endMs   = j.terminal_at ? Date.parse(j.terminal_at) : Date.now()
  if (Number.isNaN(startMs)) return null
  return Math.max(0, endMs - startMs)
}

export function formatRuntime(ms: number | null): string {
  if (ms === null) return '—'
  const total = Math.floor(ms / 1000)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}
