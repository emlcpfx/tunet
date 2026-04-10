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

export interface DbBillingEvent {
  id: string
  user_id: string
  job_id: string | null
  type: 'top_up' | 'compute_charge' | 'manual_adjustment' | 'refund'
  amount_cents: number
  description: string | null
  stripe_payment_intent: string | null
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

// ── Billing ───────────────────────────────────────────────────────────────────

export interface CreditPack {
  id: string
  label: string              // charge amount shown on button, e.g. '$10.61'
  balance_cents: number      // dollars added to user balance, in cents — same unit as price_cents
  price_cents: number        // what Stripe charges, in cents — slightly higher to cover 2.9% + $0.30
  bonus_pct?: number
}

// Price formula: ceil((balance_cents + 30) / 0.971)
// User pays price_cents, gets balance_cents added to their account.
// The difference covers the Stripe fee so operator always nets the full balance amount.
export const CREDIT_PACKS: CreditPack[] = [
  { id: 'pack_10',  label: '$10.61',  balance_cents:  1000, price_cents:  1061 },
  { id: 'pack_25',  label: '$26.06',  balance_cents:  2500, price_cents:  2606 },
  { id: 'pack_50',  label: '$51.81',  balance_cents:  5250, price_cents:  5181, bonus_pct: 5 },
  { id: 'pack_100', label: '$103.30', balance_cents: 11000, price_cents: 10330, bonus_pct: 10 },
]

export function formatCredits(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  return `${m}m`
}
