/**
 * Spark Cloud Studio compute provider.
 * Mirrors the interface of runpod.ts so compute.ts can swap providers transparently.
 *
 * OPEN GAPS — needs Walt to confirm:
 *   1. startScript / user-data: no field in current API docs.
 *      Options: (a) add a startup_script field to the deploy endpoint,
 *               (b) pre-bake a polling agent into a custom workstation preset,
 *               (c) run job_worker.py on a persistent Spark workstation that SSHes in.
 *   2. Monitor proxy URL: RunPod uses {podId}-{port}.proxy.runpod.net.
 *      Set SPARK_PROXY_BASE=https://{id}-{port}.your-domain.com or similar.
 *   3. Workstation status field names: the docs show a placeholder object —
 *      update SparkWorkstation type once Walt shares a real response sample.
 */

import { buildBootstrapScript } from './runpod'  // same bash script, different delivery
export { buildBootstrapScript }

const SPARK_API = 'https://api.prod.aapse1.sparkcloud.studio'
const MONITOR_PORT = 8080

// ── Auth — lazy JWT with auto-refresh ─────────────────────────────────────────

let _token: string | null = null
let _tokenExpiresAt = 0

function sparkCreds() {
  const email    = process.env.SPARK_EMAIL
  const password = process.env.SPARK_PASSWORD
  if (!email || !password) throw new Error('SPARK_EMAIL and SPARK_PASSWORD not set')
  return { email, password }
}

function jwtExpiry(token: string): number {
  try {
    const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString())
    return (payload.exp ?? 0) * 1000
  } catch {
    return 0
  }
}

async function getToken(): Promise<string> {
  if (_token && _tokenExpiresAt > Date.now() + 60_000) return _token

  const { email, password } = sparkCreds()
  const res = await fetch(`${SPARK_API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) throw new Error(`Spark auth failed: HTTP ${res.status}`)
  const json = await res.json()
  const tok = json.token ?? json.access_token
  if (!tok) throw new Error('Spark auth: no token in response')

  _token = tok as string
  _tokenExpiresAt = jwtExpiry(_token)
  return _token
}

async function sparkFetch<T = unknown>(
  method: 'GET' | 'POST' | 'DELETE',
  path: string,
  body?: unknown,
): Promise<T> {
  const token = await getToken()
  const res = await fetch(`${SPARK_API}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      'accept': '*/*',
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    next: { revalidate: 0 },
  })
  if (!res.ok) throw new Error(`Spark API HTTP ${res.status} on ${method} ${path}`)
  return res.json() as Promise<T>
}

// ── GPU catalog ───────────────────────────────────────────────────────────────
// Maps our gpu_type_id (stored as "spark_{staticId}") to a workstation preset.
// Use single-GPU instances for per-job cost efficiency.

export interface SparkGpuTier {
  gpu_type_id: string      // "spark_181"
  static_id: number        // 181
  display_name: string
  gpu_model: string
  vram_gb: number
  gpu_count: number
  cpu_cores: number
  ram_gb: number
}

export const SPARK_GPU_TIERS: SparkGpuTier[] = [
  { gpu_type_id: 'spark_164', static_id: 164, display_name: 'Starter — A10',     gpu_model: 'NVIDIA A10',   vram_gb: 24, gpu_count: 1, cpu_cores: 4,  ram_gb: 16  },
  { gpu_type_id: 'spark_168', static_id: 168, display_name: 'Studio — A10',      gpu_model: 'NVIDIA A10',   vram_gb: 24, gpu_count: 1, cpu_cores: 16, ram_gb: 64  },
  { gpu_type_id: 'spark_179', static_id: 179, display_name: 'Mid — L4',          gpu_model: 'NVIDIA L4',    vram_gb: 24, gpu_count: 1, cpu_cores: 8,  ram_gb: 32  },
  { gpu_type_id: 'spark_181', static_id: 181, display_name: 'Mid — L40S',        gpu_model: 'NVIDIA L40S',  vram_gb: 48, gpu_count: 1, cpu_cores: 8,  ram_gb: 64  },
  { gpu_type_id: 'spark_204', static_id: 204, display_name: 'Studio — L40S',     gpu_model: 'NVIDIA L40S',  vram_gb: 48, gpu_count: 1, cpu_cores: 16, ram_gb: 128 },
  { gpu_type_id: 'spark_205', static_id: 205, display_name: 'Pro — L40S',        gpu_model: 'NVIDIA L40S',  vram_gb: 48, gpu_count: 1, cpu_cores: 32, ram_gb: 256 },
  { gpu_type_id: 'spark_174', static_id: 174, display_name: 'Multi — 4× A10',   gpu_model: 'NVIDIA A10',   vram_gb: 24, gpu_count: 4, cpu_cores: 48, ram_gb: 192 },
  { gpu_type_id: 'spark_180', static_id: 180, display_name: 'Ludicrous — 8× L40S', gpu_model: 'NVIDIA L40S', vram_gb: 48, gpu_count: 8, cpu_cores: 192, ram_gb: 1536 },
]

export function sparkStaticId(gpuTypeId: string): number {
  const tier = SPARK_GPU_TIERS.find(t => t.gpu_type_id === gpuTypeId)
  if (tier) return tier.static_id
  // Allow raw numeric string fallback: "spark_181" → 181
  const m = gpuTypeId.match(/^spark_(\d+)$/)
  if (m) return parseInt(m[1])
  throw new Error(`Unknown Spark gpu_type_id: ${gpuTypeId}`)
}

// ── Workstation types ─────────────────────────────────────────────────────────
// Shapes confirmed from live API response.
// os_ready: -2 = stopped/deallocated, 0 = starting (unconfirmed), 1 = ready (unconfirmed)
// Ask Walt for the full os_ready lifecycle values.

export interface SparkWorkstation {
  id: number
  created_at: string
  workstation_name: string
  region: string
  workstation_static_id: number
  instance_id: string
  created_by: string
  os_ready: number          // -2 stopped, 1 running (confirm values with Walt)
  retire_date: string | null
  ami_id: string
  os_type: string
  app_type: string
  cpu: string
  gpu: string               // e.g. "NVIDIA L40S 48 GB"
  ram: string
  cost: number              // $/hr as charged by Spark
  nvme: string
  value: string
  instance_type: string
  cpu_description: string
  gpu_description: string
  instance_startup_time: number   // seconds
  os_name: string
  // TODO: ask Walt for public_ip / proxy URL fields when running
  public_ip?: string | null
}

// Shape returned by /api/workstations/{id}
interface GetWorkstationResponse {
  status: string
  data: SparkWorkstation
  error: null | string
}

// ── Queries ───────────────────────────────────────────────────────────────────

export async function listWorkstations(): Promise<SparkWorkstation[]> {
  // Real response: { status, data: { status, data: { data: [...], error } } }
  const res = await sparkFetch<{ status: string; data: { status: string; data: { data: SparkWorkstation[]; error: unknown } } }>(
    'GET', '/api/workstations'
  )
  return res?.data?.data?.data ?? []
}

export async function getWorkstation(id: string | number): Promise<SparkWorkstation | null> {
  try {
    const res = await sparkFetch<GetWorkstationResponse>('GET', `/api/workstations/${id}`)
    return res.data ?? null
  } catch {
    return null
  }
}

// ── Mutations ─────────────────────────────────────────────────────────────────

export interface CreatePodInput {
  name: string
  gpuTypeId: string
  startScript: string       // base64-encoded bash — see GAP #1 above
  containerDiskGb: number
  volumeGb: number
  env?: Record<string, string>
  region?: string
}

export interface CreatedPod {
  id: string                // Spark workstation_id as string
  name: string
  costPerHr: number         // 0 until Spark exposes per-workstation pricing
  machineId: string
}

export async function createPod(input: CreatePodInput): Promise<CreatedPod> {
  const staticId = sparkStaticId(input.gpuTypeId)
  const region   = input.region ?? process.env.SPARK_DEFAULT_REGION ?? 'us-east-1'

  // Step 1 — create temp workstation
  const temp = await sparkFetch<{ success: boolean; id: number; workstation_name: string }>(
    'POST',
    '/api/supabase/workstation/store-temp-workstation?context=app',
    {
      region,
      workstation_static_id: staticId,
      os_type:               'ubuntu24',
      app_type:              'base',
      template_version:      null,
      // GAP #1: pass startup script here once Walt adds the field
      // startup_script: input.startScript,
    },
  )

  if (!temp.success || !temp.id) throw new Error('Spark: temp workstation creation failed')

  // Step 2 — deploy
  const deployed = await sparkFetch<{ status: string; workstation_id: number }>(
    'POST',
    '/api/workstations/v2',
    { temp_workstation_id: String(temp.id) },
  )

  if (deployed.status !== 'success' || !deployed.workstation_id) {
    throw new Error('Spark: workstation deployment failed')
  }

  // TODO (GAP #1): execute startScript on the workstation once a mechanism exists.
  // Likely: POST /api/workstations/{id}/exec  or  SSH + run script in background.
  // For now, the script is stored in the job record in Supabase and must be picked up
  // by job_worker.py running on a persistent Spark workstation.

  return {
    id:        String(deployed.workstation_id),
    name:      input.name,
    costPerHr: 0,   // Spark doesn't expose real-time $/hr in this response
    machineId: String(staticId),
  }
}

export async function stopPod(workstationId: string): Promise<void> {
  await sparkFetch('POST', `/api/workstations/${workstationId}/stop`)
}

export async function startPod(workstationId: string): Promise<void> {
  await sparkFetch('POST', `/api/workstations/${workstationId}/start`)
}

export async function terminatePod(workstationId: string): Promise<void> {
  await sparkFetch('DELETE', `/api/workstations/${workstationId}`)
}

// ── Monitor proxy ─────────────────────────────────────────────────────────────
// GAP #2: RunPod exposes https://{podId}-{port}.proxy.runpod.net automatically.
// Spark proxy URL format is unknown. Set SPARK_PROXY_BASE in env once Walt confirms.
// Expected format: https://{workstation_id}-{port}.proxy.sparkcloud.studio
// (or whatever the real pattern is)

export function monitorBaseUrl(workstationId: string): string {
  const base = process.env.SPARK_PROXY_BASE ?? 'https://{id}-{port}.proxy.sparkcloud.studio'
  return base
    .replace('{id}',   workstationId)
    .replace('{port}', String(MONITOR_PORT))
}

export async function proxyMonitor<T = unknown>(
  workstationId: string,
  path: string,
  init?: RequestInit,
): Promise<{ data: T | null; error: string | null }> {
  const url = `${monitorBaseUrl(workstationId)}${path}`
  try {
    const res = await fetch(url, {
      ...init,
      signal: AbortSignal.timeout(8000),
      next: { revalidate: 0 },
    })
    if (!res.ok) return { data: null, error: `Monitor HTTP ${res.status}` }
    const data = await res.json() as T
    return { data, error: null }
  } catch (e) {
    return { data: null, error: e instanceof Error ? e.message : 'Monitor unreachable' }
  }
}

// Re-export aliases so compute.ts can present a unified surface
export { getWorkstation as getPod }
