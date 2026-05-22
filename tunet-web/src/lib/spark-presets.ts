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
    /**
     * Default for whether the preset turns on loss weighting by mask. For
     * tasks with small regions of interest (smudges, dust, paint fixes) this
     * needs to be true or the optimizer collapses to identity-function on
     * the small affected area. The user can still override via advanced.
     */
    use_mask_loss?:   boolean
    /**
     * Default mask weight when use_mask_loss (or use_auto_mask) is on. The
     * platform default is 10 which is too low for small ROIs — Beauty-class
     * presets ship with 100 so the smudge-removal use case works out of the
     * box without the user having to discover the knob.
     */
    mask_weight?:     number
  }
  // Optional behavior flags
  skip_empty_patches:    boolean
  progressive_resolution: boolean
}

export const PRESETS: Record<PresetKey, Preset> = {
  general: {
    key:         'general',
    name:        'General Image-to-Image',
    description: 'No masking — treats the whole frame equally. Reach for it when the entire image changes (a grade, an AOV, a global look), not for small spot fixes.',
    tags:        ['whole-frame', 'global looks'],
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
    description: 'Best for small, local fixes — blemishes, wire/rig removal, skin retouch. Focuses hard on just the area you changed so tiny edits aren\'t washed out, and is tuned to keep skin and texture looking real.',
    tags:        ['small fixes', 'most popular', 'auto-mask'],
    model:    { model_type: 'msrn', model_size_dims: 64, recurrence_steps: 2 },
    data:     { resolution: 512, overlap_factor: 0.5 },
    training: { loss: 'l1+lpips', lambda_lpips: 0.2, lr: 1e-4, iterations_per_epoch: 500 },
    // use_mask_loss + mask_weight=100 are the difference between this preset
    // actually fixing small smudges and the optimizer collapsing to identity.
    // With small ROIs (e.g. a lipstick smudge on a glass that's 0.3% of frame
    // area) the platform default mask_weight=10 gives the masked region only
    // ~3% of total loss — the model is rewarded more for matching the
    // unchanged background pixel-perfect than for learning the smudge. At
    // weight=100 the masked region contributes ~23% of loss, which is enough
    // to break the identity-function trap.
    mask: {
      use_auto_mask: true,
      auto_mask_gamma: 0.5,
      use_mask_loss: true,
      mask_weight: 100,
    },
    skip_empty_patches: true,
    progressive_resolution: false,
  },
  paintout: {
    key:         'paintout',
    name:        'Paintout / Cleanup',
    description: 'Best for removing large objects and cleaning big areas. Tuned for smooth, even blends across the fill, and skips the untouched parts of the frame so training goes where the work is.',
    tags:        ['large areas', 'smooth blends', 'auto-mask'],
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
    description: 'Turn a plate into a black-and-white matte. Show it plate → matte pairs and it learns to pull the matte on new frames.',
    tags:        ['matte', 'B/W output'],
    model:    { model_type: 'unet', model_size_dims: 128 },
    data:     { resolution: 512, overlap_factor: 0.25 },
    training: { loss: 'bce+dice', lambda_lpips: 0, lr: 3e-4, iterations_per_epoch: 500 },
    mask:     { use_auto_mask: false },
    skip_empty_patches: false,
    progressive_resolution: true,
  },
}

// ── User-facing options layered on top of presets ────────────────────────────

/**
 * Mirrors the full shape of `gather_config_from_ui()` in tunet.py so the web
 * form reaches parity with the desktop. Most fields are optional — when a
 * field is `undefined`, buildConfig() falls back to the preset default (or
 * the base/base.yaml default).
 */
export interface AdvancedOverrides {
  // Model
  model_size_dims?:  number
  model_type?:       'unet' | 'msrn'
  recurrence_steps?: number

  // Patches
  resolution?:       number
  overlap_factor?:   number
  // 'auto' resolves at submit time based on the picked sample's file extension
  // (.exr → linear; everything else → sRGB). buildConfig() expects this to be
  // resolved by the caller; passing 'auto' through unchanged would let it
  // leak into the YAML config that tunet doesn't understand.
  color_space?:      'auto' | 'srgb' | 'linear'

  // Optim
  loss?:             Preset['training']['loss']
  lr?:               number
  lr_scheduler?:     'none' | 'cosine' | 'plateau'
  lambda_lpips?:     number
  l1_weight?:        number
  l2_weight?:        number
  lpips_weight?:     number
  use_amp?:          boolean

  // Schedule
  batch_size?:                number   // 0 = auto-batch
  iterations_per_epoch?:      number
  max_steps?:                 number
  progressive_resolution?:    boolean

  // Mask
  use_mask_loss?:        boolean
  mask_weight?:          number
  use_mask_input?:       boolean
  use_auto_mask?:        boolean
  auto_mask_gamma?:      number
  skip_empty_patches?:   boolean
  skip_empty_threshold?: number

  // Saving
  keep_last_checkpoints?: number

  // Logging
  log_interval?:          number
  preview_batch_interval?: number
  preview_refresh_rate?:   number

  // Early stopping
  es_enabled?:  boolean
  es_patience?: number
  es_stop?:     boolean

  // Auto export
  auto_export_interval?: number
  auto_export_flame?:    boolean
  auto_export_nuke?:     boolean

  // Augmentations — each is null/undefined when disabled, an object when enabled
  augs?: AugConfig
}

export interface AugConfig {
  hflip?:  { p: number }
  affine?: {
    p: number
    scale_min: number; scale_max: number
    translate_min: number; translate_max: number
    rotate_min: number; rotate_max: number
    shear_min: number; shear_max: number
    keep_ratio: boolean
  }
  gamma?: { p: number; gamma_min: number; gamma_max: number }
  color?: {
    p: number
    brightness_min: number; brightness_max: number
    contrast_min: number; contrast_max: number
    saturation_min: number; saturation_max: number
  }
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
 * Resolve the user's color-space choice to a concrete 'srgb' | 'linear'.
 * tunet only branches on `is_linear = (color_space == 'linear')` (see
 * inference.py:132 and train.py:152), so any non-linear value collapses to
 * sRGB at the training side anyway. We pick explicitly here so the YAML
 * config is honest about what got selected.
 *
 * 'auto' picks based on the picked sample's file extension when known; if
 * `sampleExt` is null/undefined (e.g. user supplied manual ShareSync paths),
 * we fall back to sRGB since we can't see what's on disk.
 */
export function resolveColorSpace(
  choice: 'auto' | 'srgb' | 'linear' | undefined,
  sampleExt: string | null | undefined,
): 'srgb' | 'linear' {
  if (choice === 'srgb' || choice === 'linear') return choice
  // 'auto' or undefined falls through here.
  if (sampleExt && sampleExt.toLowerCase() === '.exr') return 'linear'
  return 'srgb'
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
  // Resolve common values with fallbacks
  const useAutoMask = overrides.use_auto_mask ?? preset.mask.use_auto_mask
  const loss        = overrides.loss          ?? preset.training.loss
  // Read mask settings from preset first, then override, then hardcoded
  // default — so presets that ship with mask weighting enabled (e.g. Beauty)
  // actually train with weighting on. Pre-presets-with-mask-defaults the
  // chain was `overrides ?? false / 10`, which silently ignored what the
  // preset declared.
  const useMaskLoss     = overrides.use_mask_loss     ?? preset.mask.use_mask_loss     ?? false
  const useMaskInput    = overrides.use_mask_input    ?? false
  const skipEmptyPatch  = overrides.skip_empty_patches ?? preset.skip_empty_patches
  const progressiveRes  = overrides.progressive_resolution ?? preset.progressive_resolution

  return {
    data: {
      src_dir:        inputs.src_dir,
      dst_dir:        inputs.dst_dir,
      output_dir:     inputs.output_dir,
      resolution:     overrides.resolution     ?? preset.data.resolution,
      overlap_factor: overrides.overlap_factor ?? preset.data.overlap_factor,
      color_space:    resolveColorSpace(overrides.color_space, undefined),
      mask_dir:       null,
      val_src_dir:    inputs.val_src_dir ?? null,
      val_dst_dir:    inputs.val_dst_dir ?? null,
    },
    mask: {
      use_mask_loss:        useMaskLoss,
      // Same fallback fix as use_mask_loss above — preset.mask.mask_weight
      // wins over the hardcoded 10 when the user hasn't explicitly overridden.
      mask_weight:          overrides.mask_weight ?? preset.mask.mask_weight ?? 10.0,
      use_mask_input:       useMaskInput,
      use_auto_mask:        useAutoMask,
      auto_mask_gamma:      overrides.auto_mask_gamma ?? preset.mask.auto_mask_gamma ?? 1.0,
      skip_empty_patches:   skipEmptyPatch,
      skip_empty_threshold: overrides.skip_empty_threshold ?? 3.0,
    },
    model: {
      model_size_dims:  overrides.model_size_dims ?? preset.model.model_size_dims,
      model_type:       overrides.model_type      ?? preset.model.model_type,
      ...(overrides.recurrence_steps !== undefined
          ? { recurrence_steps: overrides.recurrence_steps }
          : preset.model.recurrence_steps !== undefined
          ? { recurrence_steps: preset.model.recurrence_steps }
          : {}),
    },
    training: {
      finetune_from:        inputs.finetune_from ?? null,
      iterations_per_epoch: overrides.iterations_per_epoch ?? preset.training.iterations_per_epoch,
      batch_size:           overrides.batch_size ?? 2,   // 0 = auto in train.py
      lr:                   overrides.lr ?? preset.training.lr,
      lr_scheduler:         overrides.lr_scheduler ?? 'none',
      loss,
      lambda_lpips:         overrides.lambda_lpips ?? preset.training.lambda_lpips,
      // AMP defaults to on — matches base/base.yaml and the desktop app
      // (training_tab.py:293). Almost no quality impact and ~2× faster on
      // any Tensor-core GPU (T4 / A10 / L4 / L40S / RTX PRO). Users on
      // ancient cards without TC support won't see the speedup but won't
      // see harm either; AMP gracefully degrades.
      use_amp:              overrides.use_amp ?? true,
      ...(loss === 'weighted' ? {
        l1_weight:    overrides.l1_weight   ?? preset.training.l1_weight   ?? 1.0,
        l2_weight:    overrides.l2_weight   ?? preset.training.l2_weight   ?? 0.0,
        lpips_weight: overrides.lpips_weight ?? preset.training.lpips_weight ?? 0.1,
      } : {}),
      ...(overrides.max_steps !== undefined && overrides.max_steps > 0
          ? { max_steps: overrides.max_steps } : {}),
      ...(progressiveRes ? { progressive_resolution: true } : {}),
    },
    logging: {
      log_interval:           overrides.log_interval           ?? 5,
      // Default 50: low enough that the first preview lands within ~10-30s of
      // training starting (vs. 6+ min at the train.py default of 500). The
      // desktop app uses 35 — slightly more eager since the user is *watching*
      // there. We pick 50 because every preview save also runs forward
      // inference on the model, which costs real money on cloud GPUs.
      // Users who want fewer (or none) can override in Advanced.
      preview_batch_interval: overrides.preview_batch_interval ?? 50,
      preview_refresh_rate:   overrides.preview_refresh_rate   ?? 5,
    },
    saving: {
      keep_last_checkpoints: overrides.keep_last_checkpoints ?? 4,
    },
    early_stopping: {
      enabled:  overrides.es_enabled  ?? false,
      patience: overrides.es_patience ?? 30,
      stop:     overrides.es_stop     ?? false,
    },
    auto_export: {
      // Default to exporting ONNX every 10 epochs during training. The
      // training job's GPU already has the model loaded, so an export costs
      // ~5-10s of GPU time vs. spawning a fresh ~$0.02 / 3-min Spark job
      // later via the "Export now" button. With this on by default, the
      // Downloads panel surfaces a ready-to-use .onnx as soon as the first
      // interval lands — and the on-demand export route short-circuits to
      // the existing file when it's recent enough.
      interval: overrides.auto_export_interval ?? 10,
      flame:    overrides.auto_export_flame    ?? true,
      nuke:     overrides.auto_export_nuke     ?? false,
    },
    dataloader: {
      num_workers: -1,
      datasets: {
        shared_augs: buildAugList(overrides.augs ?? {}),
      },
    },
  }
}

/**
 * Build the albumentations shared_augs list from the form's Aug toggles.
 * Mirrors gather_config_from_ui()'s aug-construction logic in tunet.py.
 */
function buildAugList(augs: AugConfig): unknown[] {
  const out: unknown[] = []
  if (augs.hflip) {
    out.push({ _target_: 'albumentations.HorizontalFlip', p: augs.hflip.p })
  }
  if (augs.affine) {
    out.push({
      _target_:           'albumentations.Affine',
      scale:              [augs.affine.scale_min, augs.affine.scale_max],
      translate_percent:  [augs.affine.translate_min, augs.affine.translate_max],
      rotate:             [augs.affine.rotate_min, augs.affine.rotate_max],
      shear:              [augs.affine.shear_min, augs.affine.shear_max],
      interpolation:      2,
      keep_ratio:         augs.affine.keep_ratio,
      p:                  augs.affine.p,
    })
  }
  if (augs.gamma) {
    out.push({
      _target_:    'albumentations.RandomGamma',
      gamma_limit: [augs.gamma.gamma_min, augs.gamma.gamma_max],
      p:           augs.gamma.p,
    })
  }
  if (augs.color) {
    out.push({
      _target_:         'albumentations.RandomBrightnessContrast',
      brightness_limit: [augs.color.brightness_min, augs.color.brightness_max],
      contrast_limit:   [augs.color.contrast_min,   augs.color.contrast_max],
      p:                augs.color.p,
    })
    out.push({
      _target_:         'albumentations.HueSaturationValue',
      hue_shift_limit:  0,
      sat_shift_limit:  [augs.color.saturation_min, augs.color.saturation_max],
      val_shift_limit:  0,
      p:                augs.color.p,
    })
  }
  return out
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
 * Steps recommendation based on dataset size, calibrated against:
 *
 *   - learn.foundry.com `cc-train.html`: their canonical example is 10,000
 *     epochs over 6 frames with batch 4 → 15,000 steps. Sets a *floor*
 *     for tiny datasets, but Foundry's stop signal is "looks right" not
 *     a step count, so the literature underreports converged-run length.
 *
 *   - **Real-run reference (2026-04-09, RTX 3090, internal):** 96k steps
 *     to plateau on a 457-slice dataset (~30–50 frames) at 1000
 *     iterations/epoch with L1+LPIPS. Plateau detector triggered at
 *     epoch 96 with best smoothed loss @ epoch 88. This is the only
 *     converged-run data point we have, and it's *much* longer than
 *     Foundry's published examples — Foundry stops on visual match,
 *     which often happens earlier than loss plateau.
 *
 * Tier translation: the 3090 run = ~50 frames → 96k steps. Scale linearly-
 * ish with dataset size since longer datasets need more passes for the
 * model to see all variation, but cap to avoid runaway estimates on
 * very large shots (past ~200 frames the marginal value of more steps
 * drops sharply on a per-shot model).
 *
 * tunet's `max_steps=0` (run until stopped) is the usual mode, so this
 * function is purely for estimating a *typical* converged run for the
 * cost preview. Users can stop earlier when they see the result.
 */
export function recommendedStepsForPairs(pairs: number): number {
  if (pairs <= 10)   return 30_000   // floor — tiny datasets converge fast
  if (pairs <= 30)   return 60_000
  if (pairs <= 80)   return 100_000  // matches the 96k reference run
  if (pairs <= 200)  return 130_000
  return 160_000                     // long shot — diminishing returns past here
}

/**
 * Per-step time-multiplier from training settings. 1.0 = baseline UNet at
 * model_size 64, batch 2, plain L1 loss.
 *
 * **Empirically calibrated 2026-05-02** from 9 local-RTX-3090 + 9 cloud
 * cells (T4/L4/A10 at ref; L4 with torch.compile A/B). See benchmark.md.
 *
 * Calibrated parameters:
 *   - sizeMult: (dims/64)^1.47   (was 1.7^log2(dims/64) = ^0.77)
 *     fitted from porter (dim=128) running ~5.19× slower than ref on
 *     RTX 3090 — dividing out msrn=1.5× and lpips=1.25× gives dim=128
 *     ≈ 2.77×, implying exponent log₂(2.77)=1.47. dim=128 → 2.77×,
 *     dim=256 → 7.7× (capped at 8×).
 *   - msrn:   1.5×    (still theoretical — couldn't isolate from porter)
 *   - lpips:  1.25×   (still theoretical)
 *   - batch:  (bs/2)^0.59   (was √(bs/2) = ^0.5)
 *     fitted from bs=4/8 measurements: real penalty is steeper than √.
 *     bs=4 → 1.47×, bs=8 → 2.39× per-step time vs bs=2.
 */
function settingsMultiplier(opts: {
  model_size_dims?: number
  model_type?:      'unet' | 'msrn'
  loss?:            Preset['training']['loss']
  batch_size?:      number
}): number {
  // Empirical: dim=128 → 2.77× per-step on 3090. cap at 8× for absurd dims
  // (dim=512 would predict ~22×, but probably OOMs first).
  const dims = opts.model_size_dims ?? 64
  const sizeMult = Math.min(8, Math.pow(dims / 64, 1.47))

  let mult = sizeMult
  if (opts.model_type === 'msrn') mult *= 1.5
  if (opts.loss === 'l1+lpips')   mult *= 1.25
  if (opts.loss === 'weighted')   mult *= 1.30
  // bce+dice is mask-only and cheap — no penalty.

  // Empirical: bs=4 → 1.47×, bs=8 → 2.39× per-step
  const bs = opts.batch_size && opts.batch_size > 0 ? opts.batch_size : 2
  mult *= Math.pow(bs / 2, 0.59)

  return mult
}

/**
 * Per-step time-multiplier from resolution. 1.0 = 512px reference.
 *
 * **Empirically calibrated 2026-05-02.** The naive `(res/512)²` model
 * over-predicted speedup at small res by ~1.6×. Real measurements show
 * ~13% of step time is fixed overhead (kernel launches, Python loop,
 * dataloader baseline) regardless of pixel count — only the remaining
 * ~87% scales quadratically with resolution.
 *
 * Fit (12 cells across T4/L4/A10/3090):
 *   penalty(res) = 0.13 + 0.87 × (res/512)²
 *   256  → 0.35  (was 0.25)
 *   512  → 1.00
 *   1024 → 3.61  (was 4.00) — not yet measured, predicted only
 */
function resolutionPenalty(res: number): number {
  return 0.13 + 0.87 * Math.pow(res / 512, 2)
}

/**
 * Baseline steps/sec for a GPU at the reference settings (UNet, model_size
 * 64, 512px, batch 2, L1 loss). Calibration data + methodology lives in
 * benchmark.md.
 *
 * Calibration status (2026-05-02):
 *   - T4   → 4.46 step/sec  (cloud bench, real EXR — pre-torch.compile)
 *   - L4   → 7.23 step/sec  (cloud bench, real EXR — torch.compile ENABLED;
 *                              measured 6.39 with compile off, +13% with on)
 *   - A10  → 8.51 step/sec  (cloud bench, real EXR — pre-torch.compile)
 *   - L40S → guess          (g6e.8xlarge eligible, same GPU)
 *   - RTX PRO 6000 → guess  (Blackwell, g7e.2xlarge eligible)
 *
 * NOTE: torch.compile (enabled by default on cloud since commit b1ed932,
 * gated by Triton availability) gives ~13% per-step on L4. T4/A10
 * baselines were measured *before* compile was wired in and have not
 * been re-measured — production runs on those GPUs are likely 5–15%
 * faster than the numbers below. The estimator therefore over-bids
 * a bit on T4/A10/L40S/RTX PRO. Re-measure those when capacity allows.
 *
 * Local-only reference: RTX 3090 → 8.17 step/sec (matches business partner's
 * 1.62 step/s @ porter/512/bs2 to within 2% — calibration is reproducible).
 */
function baselineStepsPerSec(sku: string): number {
  if (sku.startsWith('g4dn'))      return 4.46  // T4 (measured 2026-05-02, pre-compile)
  if (sku.startsWith('g6.'))       return 7.23  // L4 (measured 2026-05-02, compile-on)
  if (sku.startsWith('g5'))        return 8.51  // A10 (measured 2026-05-02, pre-compile)
  if (sku.startsWith('g6e'))       return 12    // L40S (guess; A10 × ~1.4 from TFLOPS ratio)
  if (sku.startsWith('g7e'))       return 18    // RTX PRO 6000 Blackwell (guess)
  return 6
}

export interface EstimateRange {
  /** Most-likely runtime in hours (central point). */
  hours:            number
  /** Lower-bound (optimistic — well-tuned/converges early). */
  lowHours:         number
  /** Upper-bound (pessimistic — heavy augmentation, contention, etc.). */
  highHours:        number
  /** Steps the estimate is based on. Either user-supplied or recommended. */
  steps:            number
  /** True iff the user left max_steps=0 and we filled in a recommendation. */
  stepsAreRecommended: boolean
  /** stepsPerSec used at the central estimate. */
  centralStepsPerSec:  number
  /** Free-text breakdown shown under the "why this estimate?" disclosure. */
  basis:            string[]
}

/**
 * Bigger-lift training-time estimator. Returns a *range* with explanation,
 * not a single number, so the UI can be honest about uncertainty.
 *
 * The central point uses baselineStepsPerSec × settingsMultiplier × resPenalty;
 * low/high are 0.6×/1.7× the central. Those bounds bracket what real runs
 * actually do — convergence-driven early stops on one end, augmentation
 * heavy / data-loader-bound runs on the other.
 */
export function estimateTraining(opts: {
  sku:               string
  pairs:             number
  resolution:        number
  maxSteps?:         number
  model_size_dims?:  number
  model_type?:       'unet' | 'msrn'
  loss?:             Preset['training']['loss']
  batch_size?:       number
}): EstimateRange {
  const userSteps = opts.maxSteps && opts.maxSteps > 0 ? opts.maxSteps : 0
  const steps = userSteps > 0 ? userSteps : recommendedStepsForPairs(opts.pairs)
  const stepsAreRecommended = userSteps === 0

  const baseline = baselineStepsPerSec(opts.sku)
  const settings = settingsMultiplier(opts)
  const resPenalty = resolutionPenalty(opts.resolution)

  // Central: throughput / settings / resolution
  const centralStepsPerSec = baseline / (settings * resPenalty)
  const centralSeconds = steps / centralStepsPerSec
  const hours = centralSeconds / 3600

  // ±band: keep wide. Real runs hit 0.5–2× of any heuristic prediction.
  const lowHours  = hours * 0.6
  const highHours = hours * 1.7

  const basis: string[] = [
    `${steps.toLocaleString()} steps${stepsAreRecommended ? ` (typical for ${opts.pairs} frames; max_steps=0)` : ` (max_steps cap)`}`,
    `~${centralStepsPerSec.toFixed(1)} step/sec at these settings`,
    `${opts.resolution}px (${resPenalty.toFixed(2)}× vs 512px)`,
  ]
  if (settings > 1.05) basis.push(`settings cost ${settings.toFixed(2)}×`)

  return { hours, lowHours, highHours, steps, stepsAreRecommended, centralStepsPerSec, basis }
}

/**
 * Back-compat: existing callers that just want a single hours number get
 * the central estimate of estimateTraining(). Prefer estimateTraining()
 * for new code so the UI can show the range.
 */
export function estimateRuntimeHours(opts: {
  sku:        string
  pairs:      number
  resolution: number
  maxSteps?:  number
}): number {
  return estimateTraining(opts).hours
}

export function estimateCostUSD(opts: {
  sku:        string
  pairs:      number
  resolution: number
  maxSteps?:  number
}): number {
  return estimateRuntimeHours(opts) * pricePerHour(opts.sku)
}
