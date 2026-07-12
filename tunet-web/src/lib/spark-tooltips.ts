/**
 * Artist-friendly tooltip copy for the New Job advanced-settings form.
 *
 * Verbatim ports of the QToolTip strings in gui/training_tab.py — same
 * voice, same examples, same numbers. Eric tuned these against real
 * artist usage on the desktop app, so we keep them word-for-word rather
 * than rewriting in our own voice.
 *
 * If you tweak a default value in spark-presets.ts, also tweak the
 * corresponding "default Xx" mention here so the help stays accurate.
 */

export const TIPS = {
  // ── Model & Patches ────────────────────────────────────────────────────
  model_type:
    `Architecture type:
  'unet' — Fast, general-purpose encoder-decoder
  'msrn' — Attention + recurrence for finer detail, trains slower

Most VFX work: stick with the preset's default.`,

  model_size_dims:
    `Hidden layer width (model capacity):
  64  — Lightweight, fast training
  128 — Balanced (recommended)
  256 — Higher quality, more VRAM, slower
  512 — Maximum quality, lots of VRAM

Bigger is not always better — 128 is the sweet spot for most tasks.`,

  resolution:
    `How big of a crop the model sees during each training step.

Think of it as a magnifying glass sliding over your image:
  256 — Small window. Fast, low VRAM. Fine for simple local fixes.
  384 — Medium. Good middle ground.
  512 — Good default. Sees enough context for most VFX tasks.
  768 — More context, more VRAM. Use for paint with large blends.
  1024 — Maximum context. T4/A10 will OOM at full batch.

Higher = slower per step (quadratic in pixels).`,

  overlap_factor:
    `How much neighboring crops overlap as they slide across the image.

  0.0  — No overlap. Fastest, fewest patches. Risk of visible
          tile seams at inference. Only for quick tests.
  0.25 — Good balance (recommended). Patches overlap 25%, smooth blends.
  0.5  — More patches per image (~2× training data). Slower, sharper edges.
  0.75 — Heavy overlap. Use for large paint fixes that need seamless blends.

Higher overlap = more training time but better seam handling.`,

  color_space:
    `Must match your source image format:
  'srgb'   — Standard 8-bit images (PNG, JPEG, TIFF)
  'linear' — Scene-linear 32-bit float (EXR)

Wrong choice = blown-out highlights or muddy shadows in output.
Web app currently doesn't decode linear EXR previews — sRGB is safest.`,

  finetune_from:
    `Optional: pick an existing .pth checkpoint to start from instead of training from scratch.
Only model weights are loaded — optimizer state resets.
Leave empty to train from scratch.`,

  // ── Optimization ───────────────────────────────────────────────────────
  loss:
    `How the model measures its errors:

  'l1'        — Pixel difference. Sharp, stable. Best all-around choice.
  'l2'        — Squared difference. Smoother results, can blur detail.
  'l1+lpips'  — L1 + perceptual loss. Sharper, more 'real-looking'. Use for beauty/skin.
  'weighted'  — Custom blend of L1 + L2 + LPIPS. Power-user only.
  'bce+dice'  — Binary segmentation loss. ONLY for roto/matte (B/W output).`,

  lr:
    `How fast the model updates its weights each step.

Start with Default (1e-4) — it works for most tasks including beauty.
Go slower (5e-5) if you have lots of data and want stability.
Go faster (3e-4 / 5e-4) if your loss decreases too slowly.

If loss spikes upward and never recovers: your LR is too high.`,

  lr_scheduler:
    `Controls how the learning rate changes over time:

  'none'    — Constant learning rate (simplest, good default)
  'cosine'  — Decays then resets each cycle (good for long runs)
  'plateau' — Drops automatically when loss stops improving`,

  lambda_lpips:
    `How much the model cares about 'looking right' vs exact pixel values.
Only active when loss is 'l1+lpips'.

Low values = output matches target pixels closely but may look flat.
High values = output looks sharp/realistic but may drift from target.

0.1–0.3 is the safe range. >0.5 risks artifacts.`,

  l1_weight:
    `Weight for L1 (Mean Absolute Error) loss.
Treats all pixel errors equally. Good baseline for sharpness.
Default 1.0. Set to 0 to disable.`,

  l2_weight:
    `Weight for L2 (Mean Squared Error) loss.
Penalizes large errors more — smoother results.
Try 0.1–0.5 alongside L1 for a blend of sharp + smooth.`,

  lpips_weight:
    `Weight for LPIPS perceptual loss.
Matches structures/textures rather than raw pixels.
Keep below 0.5 to avoid artifacts. 0.1 is a safe start.`,

  use_amp:
    `Trains ~2× faster with less GPU memory. Almost no quality impact.
Disable only if you see NaN losses (very rare).`,

  // ── Schedule ───────────────────────────────────────────────────────────
  batch_size:
    `Patches per training step.
Larger = more stable gradients but more VRAM.
Reduce to 2 if you get out-of-memory errors.

'Auto' probes VRAM at training start and picks the largest that fits.`,

  iterations_per_epoch:
    `Steps before saving a checkpoint and running validation.
Lower = more frequent saves, useful for monitoring progress.
Higher = less I/O overhead, faster effective training.`,

  max_steps:
    `Total steps before auto-stopping.
0 = train until manually stopped.
Set a limit for queue items or overnight runs.`,

  progressive_resolution:
    `Start at low resolution, then scale up to full:
  Epoch 1 → 1/4 resolution (learns shapes fast)
  Epoch 2 → 1/2 resolution (medium detail)
  Epoch 3+ → full resolution (final detail pass)

Speeds up early training. Recommended for roto/matte tasks.`,

  // ── Mask & Skip-empty ──────────────────────────────────────────────────
  use_mask_loss:
    `Training loss is weighted by mask: white pixels count more,
making the model focus on those areas.

Example: if you have a face mask, the model will prioritize
getting the face right over the background.`,

  mask_weight:
    `How much more important masked (white) regions are.
10 = white pixels contribute 10× more to loss.`,

  use_mask_input:
    `Feed the mask to the model as a 4th channel alongside RGB.
The model can then learn to treat masked/unmasked areas differently.

Warning: Changes architecture — cannot toggle mid-training.`,

  use_auto_mask:
    `Automatically create masks by comparing source and target.
Areas that differ become white (important), identical areas become black.

Great for: beauty work, paint fixes, cleanup — no manual mask files needed.`,

  predict_residual:
    `Learn only the edit: model outputs a delta, composed as out = src + delta.
Untouched pixels stay identity by construction — best for local fixes
(beauty, gloves, paintouts). Stacks with auto-mask.

Off for full-frame looks and matte training.
Warning: Changes architecture — cannot toggle mid-training.`,

  auto_mask_gamma:
    `Gamma curve applied to auto-mask.
  < 1.0 — expands white coverage (e.g. 0.5 for subtle beauty work)
  > 1.0 — contracts it (tighter focus)
  1.0   — no change`,

  skip_empty_patches:
    `Skip training patches where source and target are identical.
Speeds up training when only parts of the image have changes.

Requires Auto Mask to be enabled.`,

  skip_empty_threshold:
    `Max pixel difference threshold (0–255 scale) below which a patch is skipped.
If no pixel in the crop differs by more than this, the crop is considered empty.

Increase if too many patches are being skipped.`,

  // ── Augmentations ──────────────────────────────────────────────────────
  aug_hflip:
    `Randomly flip images left-to-right.
Good for most tasks. Disable for text or directional content.`,

  aug_affine:
    `Random scale / translate / rotate / shear.
Helps the model handle slight alignment differences between source and target.`,

  aug_affine_keep_ratio:
    `Keep aspect ratio when scaling. Recommended for paired-image tasks
so source and target stay registered.`,

  aug_gamma:
    `Randomly adjust brightness curve.
Helps the model handle images with varying exposure levels.`,

  aug_color:
    `Randomly adjust brightness, contrast, and saturation.
Applied identically to source and target pairs.

Use when: your dataset has varied lighting or color grades.
Skip when: your data is already well-matched or color-critical.`,

  aug_brightness:
    `Brightness adjustment range. 0 = no change.
Negative = darker, positive = brighter. ±0.2 is a safe default.`,

  aug_contrast:
    `Contrast adjustment range. 0 = no change.
Negative = lower contrast, positive = higher contrast.`,

  aug_saturation:
    `Saturation shift range. 0 = no change.
Negative = desaturate, positive = boost color intensity.`,

  // ── Logging & Saving ───────────────────────────────────────────────────
  log_interval:
    `Print loss to the console every N steps.
Lower = noisier log, higher = cleaner.`,

  preview_batch_interval:
    `Save a preview image (source / target / model output) every N steps.
0 = disable previews. View them in the Preview panel above.`,

  preview_refresh_rate:
    `How many preview saves before the Preview panel refreshes.
Keeps the UI responsive during heavy training.`,

  keep_last_checkpoints:
    `Number of old checkpoints to keep on disk (plus the latest).
Older checkpoints are auto-deleted to save space.

Higher = more rollback options, more disk used.`,

  // ── Early stopping ─────────────────────────────────────────────────────
  es_enabled:
    `Saves a _plateau.pth checkpoint when loss stops improving.
A safety net — you can always roll back to the best model.`,

  es_patience:
    `Number of epochs with no improvement before triggering.
30 is a safe default — lower for short runs, higher for noisy data.`,

  es_stop:
    `Actually stop training when plateau is detected.
Off = just save checkpoint and keep going (recommended for most cases).`,

  // ── Auto-export ────────────────────────────────────────────────────────
  auto_export_interval:
    `Export the model every N epochs during training.
0 = disabled. Exports go to an 'exports/' subfolder.

Useful for testing intermediate models without stopping training.`,

  auto_export_flame:
    `Export an ONNX model that drops straight into:
  • Autodesk Flame's ML node
  • Adobe After Effects (Cleanplate FX, etc.)`,

  auto_export_nuke:
    `Export a TorchScript model + Nuke helper script.
Open the helper in Nuke to produce the final .cat file used by Nuke's
Inference node.`,
} as const
