import type { DbGpuPricing } from '@/types'

// Default GPU pricing — seeded into gpu_pricing table.
// Admin can override via /admin/pricing.
// Markup target: ~50% gross margin (1.505× RunPod rate, anchored to A100 at $4.50/hr)
// Formula: platform = runpod × 1.505, rounded to nearest $0.01
export const DEFAULT_GPU_PRICING: DbGpuPricing[] = [
  {
    gpu_type_id:          'NVIDIA GeForce RTX 4090',
    display_name:         'RTX 4090',
    short_key:            '4090',
    vram_gb:              24,
    runpod_cost_per_hr:   0.74,
    platform_cost_per_hr: 1.09,   // margin: $0.35 (32%)
    is_available:         true,
    tier:                 'standard',
    sort_order:           10,
  },
  {
    gpu_type_id:          'NVIDIA A40',
    display_name:         'A40',
    short_key:            'a40',
    vram_gb:              48,
    runpod_cost_per_hr:   0.79,
    platform_cost_per_hr: 1.19,   // margin: $0.40 (34%)
    is_available:         true,
    tier:                 'standard',
    sort_order:           20,
  },
  {
    gpu_type_id:          'NVIDIA L40S',
    display_name:         'L40S',
    short_key:            'l40s',
    vram_gb:              48,
    runpod_cost_per_hr:   1.14,
    platform_cost_per_hr: 1.79,   // margin: $0.65 (36%)
    is_available:         true,
    tier:                 'recommended',
    sort_order:           30,
  },
  {
    gpu_type_id:          'NVIDIA RTX PRO 6000 Blackwell Server Edition',
    display_name:         'RTX PRO 6000 Blackwell',
    short_key:            'rtxpro6000',
    vram_gb:              96,
    runpod_cost_per_hr:   1.99,
    platform_cost_per_hr: 2.99,   // margin: $1.00 (33%)
    is_available:         true,
    tier:                 'premium',
    sort_order:           40,
  },
  {
    gpu_type_id:          'NVIDIA A100 80GB PCIe',
    display_name:         'A100 80GB',
    short_key:            'a100',
    vram_gb:              80,
    runpod_cost_per_hr:   2.99,
    platform_cost_per_hr: 4.50,   // margin: $1.51 (34%)
    is_available:         true,
    tier:                 'pro',
    sort_order:           50,
  },
  {
    gpu_type_id:          'NVIDIA A100-SXM4-80GB',
    display_name:         'A100 SXM4 80GB',
    short_key:            'a100sxm',
    vram_gb:              80,
    runpod_cost_per_hr:   3.49,
    platform_cost_per_hr: 5.25,   // margin: $1.76 (34%)
    is_available:         true,
    tier:                 'pro',
    sort_order:           60,
  },
]

export function estimateCost(platformCostPerHr: number, hours: number): string {
  return `$${(platformCostPerHr * hours).toFixed(2)}`
}

// Minimum credits needed to launch a job (must cover 1 hour)
export function requiredCreditsToLaunch(platformCostPerHr: number): number {
  return Math.ceil(platformCostPerHr * 100) // cents
}

// Credits equivalent of 15 minutes — auto-stop threshold
export function autoStopThresholdCents(platformCostPerHr: number): number {
  return Math.ceil((platformCostPerHr / 4) * 100)
}
