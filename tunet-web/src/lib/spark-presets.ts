/**
 * Training presets — mirror gui/training_tab.py _apply_preset().
 *
 * Each preset bundles model + loss + LR + auto-mask choices that work well
 * for a specific VFX task. Users start from a preset and optionally tweak
 * advanced fields (model_size_dims, resolution, max_steps, etc.).
 */

export type PresetKey = 'general' | 'beauty' | 'paintout' | 'roto'

export interface Preset {
  key:         PresetKey
  name:        string
  description: string
  tags:        string[]   // small chips shown on the card
  // Defaults applied on top of base/base.yaml:
  model: {
    model_type:       'unet' | 'msrn'
    model_size_dims:  number
    recurrence_steps?: number
  }
  data: {
    resolution:       number
    overlap_factor:   number
  }
  training: {
    loss:             'l1' | 'l1+lpips' | 'weighted' | 'bce+dice'
    lambda_lpips:     number
    lr:               number
    iterations_per_epoch: number
    l1_weight?:       number
    l2_weight?:       number
    lpips_weight?:    number
  }
  mask: {
    use_auto_mask:    boolean
    auto_mask_gamma?: number
  }
  // Optional behavior flags
  skip_empty_patches:    boolean
  progressive_resolution: boolean
}

export const PRESETS: Record<PresetKey, Preset> = {
  general: {
    key:         'general',
    name:        'General Image-to-Image',
    description: 'Balanced defaults for arbitrary src→dst mapping. MSRN with L1 loss. Good baseline.',
    tags:        ['MSRN', 'L1', 'general purpose'],
    model:    { model_type: 'msrn', model_size_dims: 64, recurrence_steps: 2 },
    data:     { resolution: 512, overlap_factor: 0.5 },
    training: { loss: 'l1', lambda_lpips: 0, lr: 1e-4, iterations_per_epoch: 500 },
    mask:     { use_auto_mask: false },
    skip_empty_patches: false,
    progressive_resolution: false,
  },
  beauty: {
    key:         'beauty',
    name:        'Beauty / Paint Fix',
    description: 'Skin retouch, blemish removal, wire/rig removal, paint cleanup. Auto-mask focuses learning on changed areas. L1+LPIPS preserves fine detail.',
    tags:        ['MSRN', 'L1+LPIPS', 'Auto-Mask'],
    model:    { model_type: 'msrn', model_size_dims: 64, recurrence_steps: 2 },
    data:     { resolution: 512, overlap_factor: 0.5 },
    training: { loss: 'l1+lpips', lambda_lpips: 0.2, lr: 1e-4, iterations_per_epoch: 500 },
    mask:     { use_auto_mask: true, auto_mask_gamma: 0.5 },
    skip_empty_patches: true,
    progressive_resolution: false,
  },
  paintout: {
    key:         'paintout',
    name:        'Paintout / Cleanup',
    description: 'Heavy modification: paint-out objects, large area cleanup. Higher overlap for seamless blends. Weighted L1+L2 loss.',
    tags:        ['MSRN', 'Weighted', 'Auto-Mask'],
    model:    { model_type: 'msrn', model_size_dims: 64, recurrence_steps: 2 },
    data:     { resolution: 512, overlap_factor: 0.75 },
    training: {
      loss:             'weighted',
      lambda_lpips:     0,
      lr:               3e-4,
      iterations_per_epoch: 500,
      l1_weight:        0.5,
      l2_weight:        0.5,
      lpips_weight:     0,
    },
    mask:     { use_auto_mask: true, auto_mask_gamma: 0.5 },
    skip_empty_patches: true,
    progressive_resolution: false,
  },
  roto: {
    key:         'roto',
    name:        'Roto / Matte',
    description: 'Segmentation and matte extraction. Source = plate, destination = B/W matte. Binary output via BCE+Dice loss.',
    tags:        ['UNet', 'BCE+Dice', 'progressive'],
    model:    { model_type: 'unet', model_size_dims: 128 },
    data:     { resolution: 512, overlap_factor: 0.25 },
    training: { loss: 'bce+dice', lambda_lpips: 0, lr: 3e-4, iterations_per_epoch: 500 },
    mask:     { use_auto_mask: false },
    skip_empty_patches: false,
    progressive_resolution: true,
  },
}

// ── User-facing options layered on top of presets ────────────────────────────

export interface AdvancedOverrides {
  model_size_dims?: number
  resolution?:      number
  batch_size?:      number
  max_steps?:       number
  loss?:            Preset['training']['loss']
  lr?:              number
  progressive_resolution?: boolean
}

export interface JobInputs {
  src_dir:        string         // e.g. '/input/data/src'
  dst_dir:        string         // e.g. '/input/data/dst'
  output_dir:     string         // e.g. '/output/<job-name>'
  val_src_dir?:   string | null
  val_dst_dir?:   string | null
  finetune_from?: string | null  // path to .pth (relative to /input)
}

/**
 * Build the synthesized config dict from a preset + user inputs + overrides.
 * Result is plain JSON, ready to YAML-dump and ship in the input tarball.
 *
 * Mirrors the structure of base/base.yaml.
 */
export function buildConfig(
  preset:    Preset,
  inputs:    JobInputs,
  overrides: AdvancedOverrides = {},
): Record<string, unknown> {
  return {
    data: {
      src_dir:        inputs.src_dir,
      dst_dir:        inputs.dst_dir,
      output_dir:     inputs.output_dir,
      resolution:     overrides.resolution ?? preset.data.resolution,
      overlap_factor: preset.data.overlap_factor,
      color_space:    'srgb',
      mask_dir:       null,
      val_src_dir:    inputs.val_src_dir ?? null,
      val_dst_dir:    inputs.val_dst_dir ?? null,
    },
    mask: {
      use_mask_loss:   false,
      mask_weight:     10.0,
      use_mask_input:  false,
      use_auto_mask:   preset.mask.use_auto_mask,
      auto_mask_gamma: preset.mask.auto_mask_gamma ?? 1.0,
    },
    model: {
      model_size_dims:  overrides.model_size_dims ?? preset.model.model_size_dims,
      model_type:       preset.model.model_type,
      ...(preset.model.recurrence_steps !== undefined ? {
        recurrence_steps: preset.model.recurrence_steps,
      } : {}),
    },
    training: {
      finetune_from:        inputs.finetune_from ?? null,
      iterations_per_epoch: preset.training.iterations_per_epoch,
      batch_size:           overrides.batch_size ?? 2,
      lr:                   overrides.lr ?? preset.training.lr,
      lr_scheduler:         'none',
      loss:                 overrides.loss ?? preset.training.loss,
      lambda_lpips:         preset.training.lambda_lpips,
      use_amp:              false,
      ...(preset.training.l1_weight   !== undefined ? { l1_weight:   preset.training.l1_weight   } : {}),
      ...(preset.training.l2_weight   !== undefined ? { l2_weight:   preset.training.l2_weight   } : {}),
      ...(preset.training.lpips_weight !== undefined ? { lpips_weight: preset.training.lpips_weight } : {}),
      ...(overrides.max_steps !== undefined && overrides.max_steps > 0 ? { max_steps: overrides.max_steps } : {}),
      ...(overrides.progressive_resolution !== undefined
        ? { progressive_resolution: overrides.progressive_resolution }
        : preset.progressive_resolution
        ? { progressive_resolution: true }
        : {}),
    },
    logging: {
      preview_batch_interval: 0,   // disabled by default; presets can re-enable later
    },
  }
}

// ── Cost estimate ────────────────────────────────────────────────────────────

/**
 * Hourly $/hr for each Spark instance (best-known pricing as of mid-2026).
 * Source: real workstation API responses + Spark public pricing page.
 * These are approximations — Spark's real billing is per-second on running.
 */
export const GPU_PRICING_USD_PER_HR: Record<string, number> = {
  'g4dn.xlarge':    0.74,    // 1× T4 16GB
  'g5.xlarge':      1.65,    // 1× A10 24GB
  'g5.2xlarge':     1.84,
  'g5.4xlarge':     2.43,
  'g5.8xlarge':     3.61,
  'g5.16xlarge':    5.99,
  'g5.24xlarge':    8.18,    // 4× A10 24GB
  'g6.2xlarge':     1.32,    // 1× L4 24GB
  'g6.4xlarge':     1.79,
  'g6.8xlarge':     2.74,
  'g6.16xlarge':    4.66,
  'g6e.xlarge':     2.85,
  'g6e.2xlarge':    3.32,
  'g6e.4xlarge':    4.99,    // 1× L40S 48GB (recommended)
  'g6e.8xlarge':    8.32,
  'g6e.12xlarge':   13.10,   // 4× L40S 48GB
  'g6e.16xlarge':   12.68,
  'g7e.xlarge':     6.50,    // 1× RTX PRO 6000 96GB (estimate)
  'g7e.2xlarge':    8.50,
  'g7e.4xlarge':    10.50,
  'g7e.48xlarge':   58.00,   // 8× RTX PRO 6000 (estimate)
}

export function pricePerHour(sku: string): number {
  return GPU_PRICING_USD_PER_HR[sku] ?? 0
}

/**
 * Rough wall-clock estimate for a training run. Uses crude T/Step assumptions
 * by GPU. The user can override `estimated_hours` in the form.
 *
 * Returns hours.
 */
export function estimateRuntimeHours(opts: {
  sku:        string
  pairs:      number       // training pairs (frame count)
  resolution: number       // 256/512/768/1024
  maxSteps?:  number       // 0/undefined = run until stopped
}): number {
  // Assume ~1500 steps per epoch (rough), 3-8 step/sec depending on GPU.
  // Without max_steps, default to 15k steps for the estimate.
  const steps = opts.maxSteps && opts.maxSteps > 0 ? opts.maxSteps : 15000

  // step/sec heuristic by GPU class (very rough)
  let stepsPerSec = 4
  if (opts.sku.startsWith('g4dn'))      stepsPerSec = 2     // T4
  else if (opts.sku.startsWith('g5'))   stepsPerSec = 4     // A10
  else if (opts.sku.startsWith('g6.'))  stepsPerSec = 5     // L4
  else if (opts.sku.startsWith('g6e'))  stepsPerSec = 8     // L40S
  else if (opts.sku.startsWith('g7e'))  stepsPerSec = 12    // RTX PRO 6000

  // Resolution penalty (512 → 1.0×, 1024 → 4.0×)
  const resPenalty = (opts.resolution / 512) ** 2

  const seconds = (steps * resPenalty) / stepsPerSec
  return seconds / 3600
}

export function estimateCostUSD(opts: {
  sku:        string
  pairs:      number
  resolution: number
  maxSteps?:  number
}): number {
  return estimateRuntimeHours(opts) * pricePerHour(opts.sku)
}
