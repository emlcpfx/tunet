// ── Job status ────────────────────────────────────────────────────────────────

export type JobStatus =
  | 'pending'
  | 'provisioning'
  | 'running'
  | 'stopped'
  | 'failed'
  | 'terminated'

// ── Database row types ────────────────────────────────────────────────────────

export interface DbUser {
  id: string
  email: string
  name: string | null
  credit_balance_cents: number
  is_admin: boolean
  created_at: string
}

export interface DbJob {
  id: string
  user_id: string
  name: string
  pod_id: string | null
  status: JobStatus
  gpu_type_id: string
  gpu_display_name: string | null
  runpod_cost_per_hr: number | null
  platform_cost_per_hr: number | null
  accumulated_cost_cents: number
  billing_last_tick_at: string | null
  config_path: string | null
  src_zip_path: string | null
  dst_zip_path: string | null
  checkpoint_path: string | null
  container_disk_gb: number
  volume_gb: number
  started_at: string | null
  ended_at: string | null
  created_at: string
}

export interface DbGpuPricing {
  gpu_type_id: string
  display_name: string
  short_key: string
  vram_gb: number | null
  platform_cost_per_hr: number
  runpod_cost_per_hr: number
  is_available: boolean
  tier: 'standard' | 'recommended' | 'pro' | 'premium'
  sort_order: number
}

// ── API response types ────────────────────────────────────────────────────────

export interface JobWithUser extends DbJob {
  user?: DbUser
}

export interface JobCreatePayload {
  name: string
  gpu_type_id: string
  config_path: string
  src_zip_path?: string
  dst_zip_path?: string
  checkpoint_path?: string
  container_disk_gb: number
  volume_gb: number
}

// ── Compute provider ─────────────────────────────────────────────────────────

export type ComputeProvider = 'runpod' | 'spark'

// ── RunPod types ──────────────────────────────────────────────────────────────

export interface RunPodPort {
  ip: string
  isIpPublic: boolean
  privatePort: number
  publicPort: number
  type: string
}

export interface RunPodGpu {
  id: string
  gpuUtilPercent: number | null
}

export interface RunPodPod {
  id: string
  name: string
  desiredStatus: 'RUNNING' | 'EXITED' | 'DEAD' | 'FAILED'
  costPerHr: number
  gpuCount: number
  imageName: string
  uptimeSeconds: number
  lastStartedAt: string
  runtime: {
    ports: RunPodPort[]
    gpus: RunPodGpu[]
  } | null
  machine: {
    gpuDisplayName: string
  } | null
}

// ── Monitor API types (from monitor_api.py) ───────────────────────────────────

export interface MonitorStatus {
  running: boolean
  step: number
  loss: number | null
  loss_label: string
  best_loss: number | null
  best_step: number
  elapsed_s: number
  elapsed: string
  step_time_s: number | null
  has_lpips: boolean
  has_val: boolean
  stop_file: boolean
}

export interface MonitorMetrics {
  steps: number[]
  l1: number[]
  loss_label: string
  lpips?: number[]
  val_steps?: number[]
  val_l1?: number[]
  val_lpips?: number[]
  val_psnr?: number[]
  val_ssim?: number[]
}

export interface MonitorLogs {
  lines: string[]
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  return `${m}m`
}
