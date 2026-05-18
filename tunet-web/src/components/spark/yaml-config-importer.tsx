'use client'

/**
 * YAML config importer for the new-job form.
 *
 * Mirrors the old "load model.yaml from disk" workflow that pre-dated the
 * preset-driven UI: pick a tunet-style YAML config, parse it client-side,
 * and pre-fill the form's Advanced Settings with whatever the YAML carries.
 *
 * Best-effort mapping: only fields with a 1:1 form equivalent are applied.
 * Spark-specific concerns (GPU choice, compute mode, idle hold, alerts,
 * ShareSync paths) aren't in tunet's YAML and stay at whatever the user
 * already had selected. Preset key isn't inferable from raw YAML either —
 * the form's preset selection isn't touched.
 */

import { useRef, useState } from 'react'
import { parse as parseYaml } from 'yaml'
import type { AdvancedOverrides } from '@/lib/spark-presets'

export interface YamlImportResult {
  advanced:        Partial<AdvancedOverrides>
  /** training.max_steps from the YAML — applied to the form's top-level Max Steps input. */
  maxSteps?:       number
  /** Filename the user picked, for the "✓ Loaded from X" badge. */
  filename:        string
  /** Top-level YAML keys we recognized + applied (for the loaded-from message). */
  appliedSections: string[]
  /** Top-level keys present in the YAML that we didn't map (data.src_dir etc.). */
  ignoredKeys:     string[]
}

interface Props {
  onLoaded: (result: YamlImportResult) => void
}

export function YamlConfigImporter({ onLoaded }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [lastLoaded, setLastLoaded] = useState<YamlImportResult | null>(null)
  const [error, setError]           = useState<string | null>(null)

  async function handleFile(file: File) {
    setError(null)
    try {
      const text = await file.text()
      const parsed = parseYaml(text)
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('YAML is empty or not an object at the top level')
      }
      const result = mapYamlToForm(parsed as Record<string, unknown>, file.name)
      setLastLoaded(result)
      onLoaded(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to parse YAML')
      setLastLoaded(null)
    }
  }

  return (
    <div className="mt-3 flex items-start gap-3 flex-wrap">
      <input
        ref={inputRef}
        type="file"
        accept=".yaml,.yml,application/x-yaml,text/yaml"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) handleFile(f)
          // Reset input so picking the same file twice still triggers onChange
          if (inputRef.current) inputRef.current.value = ''
        }}
        className="hidden"
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="px-3 py-1.5 text-xs font-semibold border border-[#e5e7eb] text-[#374151] rounded hover:bg-[#F9FAFB] flex items-center gap-1.5"
        title="Pre-fill Advanced Settings from a tunet model.yaml on your disk."
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        {lastLoaded ? 'Load a different YAML…' : 'Load settings from YAML…'}
      </button>

      {lastLoaded && (
        <div className="text-[11px] text-[#6b7280] leading-snug pt-0.5">
          <span className="text-[#16A34A] font-semibold">✓ Loaded {lastLoaded.filename}</span>
          {lastLoaded.appliedSections.length > 0 && (
            <>
              {' — applied: '}
              <code className="bg-[#F9FAFB] px-1 rounded text-[#374151]">
                {lastLoaded.appliedSections.join(', ')}
              </code>
            </>
          )}
          {lastLoaded.ignoredKeys.length > 0 && (
            <span className="text-[#9ca3af]">
              {'  ·  ignored: '}
              <code className="bg-[#F9FAFB] px-1 rounded">
                {lastLoaded.ignoredKeys.join(', ')}
              </code>
            </span>
          )}
          <p className="mt-0.5">
            Review the GPU + Advanced Settings below before submitting — the YAML
            can&apos;t set Spark-specific fields (GPU, compute mode, ShareSync paths).
          </p>
        </div>
      )}

      {error && (
        <p className="text-xs text-[#EF4444]">YAML import failed: {error}</p>
      )}
    </div>
  )
}

// ── Mapping ────────────────────────────────────────────────────────────────

/**
 * Translate a parsed tunet YAML into form-state assignments. We only pick up
 * fields that exist in AdvancedOverrides — anything else is reported in
 * ignoredKeys so the user can tell what the importer skipped.
 */
function mapYamlToForm(yaml: Record<string, unknown>, filename: string): YamlImportResult {
  const advanced: Partial<AdvancedOverrides> = {}
  const applied: string[] = []
  const ignored: string[] = []
  let maxSteps: number | undefined

  // ── data ─────────────────────────────────────────────────────────────────
  const data = obj(yaml.data)
  if (data) {
    if (typeof data.resolution === 'number')     advanced.resolution     = data.resolution
    if (typeof data.overlap_factor === 'number') advanced.overlap_factor = data.overlap_factor
    if (typeof data.color_space === 'string'
        && (data.color_space === 'srgb' || data.color_space === 'linear' || data.color_space === 'auto')) {
      advanced.color_space = data.color_space
    }
    // src_dir/dst_dir/output_dir don't translate to Spark paths — ignored.
    applied.push('data')
  }

  // ── model ────────────────────────────────────────────────────────────────
  const model = obj(yaml.model)
  if (model) {
    if (model.model_type === 'unet' || model.model_type === 'msrn') {
      advanced.model_type = model.model_type
    }
    if (typeof model.model_size_dims === 'number') advanced.model_size_dims = model.model_size_dims
    applied.push('model')
  }

  // ── training ─────────────────────────────────────────────────────────────
  const training = obj(yaml.training)
  if (training) {
    if (typeof training.batch_size === 'number')   advanced.batch_size   = training.batch_size
    if (typeof training.lr === 'number')           advanced.lr           = training.lr
    if (typeof training.lambda_lpips === 'number') advanced.lambda_lpips = training.lambda_lpips
    if (typeof training.l1_weight === 'number')    advanced.l1_weight    = training.l1_weight
    if (typeof training.l2_weight === 'number')    advanced.l2_weight    = training.l2_weight
    if (typeof training.lpips_weight === 'number') advanced.lpips_weight = training.lpips_weight
    if (training.loss === 'l1' || training.loss === 'l1+lpips'
        || training.loss === 'weighted' || training.loss === 'bce+dice') {
      advanced.loss = training.loss
    }
    if (typeof training.max_steps === 'number')    maxSteps              = training.max_steps
    applied.push('training')
  }

  // ── mask ─────────────────────────────────────────────────────────────────
  const mask = obj(yaml.mask)
  if (mask) {
    if (typeof mask.use_mask_loss === 'boolean')    advanced.use_mask_loss    = mask.use_mask_loss
    if (typeof mask.mask_weight === 'number')       advanced.mask_weight      = mask.mask_weight
    if (typeof mask.use_mask_input === 'boolean')   advanced.use_mask_input   = mask.use_mask_input
    if (typeof mask.use_auto_mask === 'boolean')    advanced.use_auto_mask    = mask.use_auto_mask
    if (typeof mask.auto_mask_gamma === 'number')   advanced.auto_mask_gamma  = mask.auto_mask_gamma
    if (typeof mask.skip_empty_patches === 'boolean') advanced.skip_empty_patches  = mask.skip_empty_patches
    if (typeof mask.skip_empty_threshold === 'number') advanced.skip_empty_threshold = mask.skip_empty_threshold
    applied.push('mask')
  }

  // ── progressive res lives at top-level training in some configs ──────────
  if (training && typeof training.progressive_resolution === 'boolean') {
    advanced.progressive_resolution = training.progressive_resolution
  } else if (typeof yaml.progressive_resolution === 'boolean') {
    advanced.progressive_resolution = yaml.progressive_resolution
  }

  // Collect top-level keys that exist but weren't recognized by any section
  // handler. Helps the user spot typos / pruned-from-old-schema fields.
  const recognized = new Set(['data', 'model', 'training', 'mask', 'progressive_resolution',
                              'dataloader', 'saving', 'logging', 'augmentation', 'preview'])
  for (const k of Object.keys(yaml)) {
    if (!recognized.has(k)) ignored.push(k)
  }

  return { advanced, maxSteps, filename, appliedSections: applied, ignoredKeys: ignored }
}

function obj(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : null
}
