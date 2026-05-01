/**
 * Client-safe Spark types — no `server-only` import, safe to use in client components.
 * Mirrors the shapes in lib/spark.ts (which is server-only).
 */

export interface SparkJob {
  id:                       string
  organisation_id?:         number
  user_id?:                 number
  availability_region?:     string
  mode?:                    string
  image?:                   string
  command?:                 string[]
  env?:                     Record<string, string>
  output_share_sync_path?:  string
  instance_type_sku_id?:    number
  instance_type_name?:      string
  status:                   string
  error_code?:              string | null
  error_message?:           string | null
  exit_code?:               number | null
  created_at?:              string
  started_provisioning_at?: string | null
  started_running_at?:      string | null
  terminal_at?:             string | null
  cancel_requested_at?:     string | null
  last_agent_heartbeat_at?: string | null
  cuda_version?:            string
  driver_version?:          string
  gpu_name?:                string
  log_archive_share_sync_path?: string
  log_archive_uploaded_at?: string | null
  idle_hold_seconds?:       number
  input_share_sync_path?:   string
  name?:                    string
}

export const ACTIVE_STATUSES = new Set(['queued', 'provisioning', 'running'])

export function isActive(j: SparkJob): boolean {
  return ACTIVE_STATUSES.has(j.status)
}

/**
 * Status to display in the UI. Spark's job-level `status` can lag behind the
 * actual container state — it stays `provisioning` until `started_running_at`
 * is set on a separate codepath, even though the container is already up.
 *
 * Resolution order:
 *   1. status !== 'provisioning'             → trust it as-is
 *   2. status === 'provisioning' AND started_running_at set
 *                                             → 'running' (canonical signal)
 *   3. status === 'provisioning' AND job has been provisioning past the
 *      typical pull window AND we've seen a heartbeat at any point
 *                                             → 'running' (inferred — Spark's
 *                                                image-pull cap is <3 min cold;
 *                                                past that, if the agent ever
 *                                                came up, the container is up.
 *                                                heartbeat freshness was too
 *                                                strict and missed jobs whose
 *                                                heartbeat field stopped
 *                                                updating once running)
 */
export function derivedStatus(j: SparkJob): string {
  if (j.status !== 'provisioning') return j.status
  if (j.started_running_at) return 'running'

  const heartbeat = j.last_agent_heartbeat_at ? Date.parse(j.last_agent_heartbeat_at) : 0
  const provStart = j.started_provisioning_at ? Date.parse(j.started_provisioning_at) : 0
  const now = Date.now()

  if (provStart) {
    const provisioningFor = now - provStart

    // Heartbeat present + 5 min elapsed → definitely running.
    if (provisioningFor > 5 * 60_000 && heartbeat) {
      return 'running'
    }

    // Spark sometimes never sets `last_agent_heartbeat_at` even after the
    // container is up and emitting logs. The provisioning phase tops out at
    // ~3-4 min in practice (image pull is the slow part); past 8 min with no
    // explicit heartbeat we assume it's running. Conservative threshold so
    // genuinely-stuck jobs still show provisioning.
    if (provisioningFor > 8 * 60_000) {
      return 'running'
    }
  }
  return j.status
}

export function jobLabel(j: SparkJob): string {
  // Spark API doesn't persist or return the submit `name`, so we resolve it
  // from a few fallbacks in priority order:
  //   1. j.name                       (would only be there if API starts echoing it)
  //   2. j.env.TUNET_JOB_NAME         (we stash the friendly name on submit)
  //   3. basename of /output/<x> in command[2]  (sanitized but still meaningful)
  //   4. first 8 chars of the job id  (last resort)
  if (j.name) return j.name

  const envName = j.env?.TUNET_JOB_NAME
  if (envName) return envName

  // Try to recover from the command we sent: ['bash', 'spark_start.sh', '/output/<jobname>', ...]
  const cmd = j.command
  if (cmd && Array.isArray(cmd) && cmd.length >= 3) {
    const outputArg = cmd[2]
    if (typeof outputArg === 'string' && outputArg.startsWith('/output/')) {
      const base = outputArg.slice('/output/'.length).split('/')[0]
      if (base) return base
    }
  }

  return j.id.length > 8 ? j.id.slice(0, 8) : j.id
}

export function jobPresetKey(j: SparkJob): string | null {
  return j.env?.TUNET_PRESET ?? null
}

export function jobGpuKey(j: SparkJob): string | null {
  return j.env?.TUNET_GPU ?? null
}

export type JobModeInfo =
  | { mode: 'new' }
  | { mode: 'resume' | 'finetune'; sourceJobId: string | null; checkpointName: string | null }

/**
 * Recover whether a job was submitted as new / resume / fine-tune from its
 * env stash. Older jobs (pre-mode env vars) all return 'new'.
 */
export function jobMode(j: SparkJob): JobModeInfo {
  const m = j.env?.TUNET_MODE
  if (m === 'resume' || m === 'finetune') {
    return {
      mode:           m,
      sourceJobId:    j.env?.TUNET_SOURCE_JOB_ID    ?? null,
      checkpointName: j.env?.TUNET_SOURCE_CHECKPOINT ?? null,
    }
  }
  return { mode: 'new' }
}

/**
 * Display name for a job's GPU. Spark's API leaves `gpu_name` empty during
 * provisioning (and on jobs that never started), but we still know the
 * GPU because we either:
 *   1. Stashed the GPU shortcut in `env.TUNET_GPU` at submit time, or
 *   2. Have the instance_type_name SKU and can look up the family.
 *
 * Returns '—' if we have nothing usable.
 *
 * Mirror of GPU_TYPES in lib/spark.ts (client-safe duplicate to avoid
 * pulling the server-only module into client components).
 */
const GPU_KEY_DISPLAY: Record<string, string> = {
  t4:           'NVIDIA T4',
  a10:          'NVIDIA A10',
  l4:           'NVIDIA L4',
  l40s:         'NVIDIA L40S',
  a10x4:        '4× NVIDIA A10',
  l40sx4:       '4× NVIDIA L40S',
  rtxpro6000:   'NVIDIA RTX PRO 6000',
  rtxpro6000x8: '8× NVIDIA RTX PRO 6000',
}

const SKU_DISPLAY: Record<string, string> = {
  'g4dn.xlarge':  'NVIDIA T4',
  'g5.xlarge':    'NVIDIA A10',
  'g6.2xlarge':   'NVIDIA L4',
  'g6e.4xlarge':  'NVIDIA L40S',
  'g5.24xlarge':  '4× NVIDIA A10',
  'g6e.12xlarge': '4× NVIDIA L40S',
  'g7e.xlarge':   'NVIDIA RTX PRO 6000',
  'g7e.48xlarge': '8× NVIDIA RTX PRO 6000',
}

export function jobGpuDisplay(j: SparkJob): string {
  if (j.gpu_name) return j.gpu_name
  const key = jobGpuKey(j)
  if (key && GPU_KEY_DISPLAY[key]) return GPU_KEY_DISPLAY[key]
  if (j.instance_type_name && SKU_DISPLAY[j.instance_type_name]) return SKU_DISPLAY[j.instance_type_name]
  return '—'
}

export function jobRuntimeMs(j: SparkJob, now: number = Date.now()): number | null {
  const start = j.started_running_at ?? j.started_provisioning_at ?? j.created_at
  if (!start) return null
  const startMs = Date.parse(start)
  const endMs   = j.terminal_at ? Date.parse(j.terminal_at) : now
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

export function formatStarted(j: SparkJob): string {
  const ts = j.started_running_at ?? j.started_provisioning_at ?? j.created_at
  if (!ts) return '—'
  try {
    const d = new Date(ts)
    return d.toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return '—'
  }
}
