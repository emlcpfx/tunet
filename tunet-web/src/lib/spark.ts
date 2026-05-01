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
  l40s:         { sku: 'g6e.4xlarge',   gpu: 'NVIDIA L40S',         vramGb: 48, gpuCount: 1, label: 'L40S 48GB · recommended' },
  a10x4:        { sku: 'g5.24xlarge',   gpu: 'NVIDIA A10',          vramGb: 24, gpuCount: 4, label: '4× A10 24GB · multi-GPU' },
  l40sx4:       { sku: 'g6e.12xlarge',  gpu: 'NVIDIA L40S',         vramGb: 48, gpuCount: 4, label: '4× L40S 48GB · multi-GPU' },
  rtxpro6000:   { sku: 'g7e.xlarge',    gpu: 'NVIDIA RTX PRO 6000', vramGb: 96, gpuCount: 1, label: 'RTX PRO 6000 96GB · fastest' },
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
  availability_region?:        string
  mode?:                       string
  image?:                      string
  command?:                    string[]
  env?:                        Record<string, string>
  output_share_sync_path?:     string
  instance_type_sku_id?:       number
  instance_type_name?:         string
  status:                      'queued' | 'provisioning' | 'running' | 'completed' | 'failed' | 'cancelled' | string
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

export async function submitJob(input: SubmitJobInput): Promise<SubmitJobResponse> {
  return sparkFetch<SubmitJobResponse>('POST', '/api/compute/jobs', {
    name:            input.name,
    instanceType:    input.instanceType,
    image:           input.image ?? DEFAULT_IMAGE,
    command:         input.command,
    inputPushMode:   'auto-prepare',
    idleHoldSeconds: input.idleHoldSeconds ?? 0,
    env:             input.env ?? {},
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
