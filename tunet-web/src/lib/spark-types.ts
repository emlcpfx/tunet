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
