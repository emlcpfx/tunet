'use client'

/**
 * Full advanced-settings panel for the New Job form. Mirrors the field set in
 * gui/training_tab.py (gather_config_from_ui) so the web form has parity with
 * the desktop app.
 *
 * Sections (each collapsible, like the desktop CollapsibleGroupBox):
 *   • Model & Patches
 *   • Optimization
 *   • Schedule
 *   • Mask & Skip-empty
 *   • Augmentations
 *   • Logging
 *   • Saving / Early stopping / Auto-export
 *
 * State is held by the parent — this component just renders + emits change
 * events. Pass `value` (current AdvancedOverrides) and `onChange` (full
 * replacement). Use `presetDefaults` so the Reset buttons can wipe back to
 * preset-baseline values rather than hardcoded constants.
 */

import { useState } from 'react'
import { FormRow, Input, InfoTip } from '@/components/ui/input'
import { resolveColorSpace, type AdvancedOverrides, type AugConfig, type Preset } from '@/lib/spark-presets'
import { TIPS } from '@/lib/spark-tooltips'

const RESOLUTIONS = [256, 384, 512, 768, 1024]
const MODEL_SIZES = [32, 64, 128, 256, 512]
const BATCH_SIZES = [0, 1, 2, 4, 8, 16]   // 0 = auto
const OVERLAPS    = [0, 0.25, 0.5, 0.75]
const LOSS_OPTS: { value: NonNullable<AdvancedOverrides['loss']>; label: string }[] = [
  { value: 'l1',       label: 'L1' },
  { value: 'l1+lpips', label: 'L1 + LPIPS' },
  { value: 'weighted', label: 'Weighted (L1 + L2 + LPIPS)' },
  { value: 'bce+dice', label: 'BCE + Dice' },
]
const LR_OPTS = [
  { value: 1e-5, label: '1e-5 (very fine)' },
  { value: 5e-5, label: '5e-5' },
  { value: 1e-4, label: '1e-4 (default)' },
  { value: 3e-4, label: '3e-4' },
  { value: 5e-4, label: '5e-4' },
  { value: 1e-3, label: '1e-3 (aggressive)' },
]

interface Props {
  value:           AdvancedOverrides
  onChange:        (v: AdvancedOverrides) => void
  preset:          Preset
  /** First src image's extension (e.g. '.exr'). Used to surface what
   *  color_space='auto' will resolve to. */
  sampleExt?:      string | null
  /** Open the Preview Filter dialog. Rendered as a button next to the
   *  skip threshold so the user can iterate without leaving Advanced. */
  onPreviewFilter?: () => void
  /** Disable the Preview Filter button when no folder is picked yet. */
  previewFilterReady?: boolean
  /** Open the Auto-Mask Preview dialog. Rendered next to the gamma input. */
  onAutoMaskPreview?: () => void
  /** Disabled until a folder is picked. */
  autoMaskPreviewReady?: boolean
}

export function AdvancedSettings({
  value, onChange, preset, sampleExt,
  onPreviewFilter, previewFilterReady,
  onAutoMaskPreview, autoMaskPreviewReady,
}: Props) {
  const set = <K extends keyof AdvancedOverrides>(k: K, v: AdvancedOverrides[K]) =>
    onChange({ ...value, [k]: v })

  return (
    <div className="space-y-2">
      <Section title="Model & Patches" defaultOpen>
        <Grid2>
          <FormRow label="Model type" tip={TIPS.model_type}>
            <Select
              value={value.model_type ?? preset.model.model_type}
              onChange={(v) => set('model_type', v as 'unet' | 'msrn')}
              options={[
                { value: 'unet', label: 'UNet (standard)' },
                { value: 'msrn', label: 'MSRN (recurrent + attention)' },
              ]}
            />
          </FormRow>
          <FormRow label="Model size (capacity)" tip={TIPS.model_size_dims}>
            <Select
              value={String(value.model_size_dims ?? preset.model.model_size_dims)}
              onChange={(v) => set('model_size_dims', parseInt(v, 10))}
              options={MODEL_SIZES.map(s => ({ value: String(s), label: String(s) }))}
            />
          </FormRow>
          <FormRow label="Resolution (px)" tip={TIPS.resolution}>
            <Select
              value={String(value.resolution ?? preset.data.resolution)}
              onChange={(v) => set('resolution', parseInt(v, 10))}
              options={RESOLUTIONS.map(r => ({ value: String(r), label: `${r}px` }))}
            />
          </FormRow>
          <FormRow label="Overlap factor" tip={TIPS.overlap_factor}>
            <Select
              value={String(value.overlap_factor ?? preset.data.overlap_factor)}
              onChange={(v) => set('overlap_factor', parseFloat(v))}
              options={OVERLAPS.map(o => ({ value: String(o), label: `${(o * 100).toFixed(0)}%` }))}
            />
          </FormRow>
          <FormRow label="Color space" tip={TIPS.color_space}>
            <div>
              <Select
                value={value.color_space ?? 'auto'}
                onChange={(v) => set('color_space', v as 'auto' | 'srgb' | 'linear')}
                options={[
                  { value: 'auto',   label: sampleExt
                      ? `Auto (${resolveColorSpace('auto', sampleExt)} from ${sampleExt.toUpperCase().slice(1)})`
                      : 'Auto (from file format)' },
                  { value: 'srgb',   label: 'sRGB' },
                  { value: 'linear', label: 'Linear (EXR)' },
                ]}
              />
              {(value.color_space ?? 'auto') === 'auto' && (
                <p className="text-[11px] text-[#9ca3af] mt-1">
                  EXR files train as linear, everything else as sRGB. Override above if your data needs it.
                </p>
              )}
            </div>
          </FormRow>
        </Grid2>
      </Section>

      <Section title="Optimization">
        <Grid2>
          <FormRow label="Loss function" tip={TIPS.loss}>
            <Select
              value={value.loss ?? preset.training.loss}
              onChange={(v) => set('loss', v as AdvancedOverrides['loss'])}
              options={LOSS_OPTS.map(l => ({ value: l.value!, label: l.label }))}
            />
          </FormRow>
          <FormRow label="Learning rate" tip={TIPS.lr}>
            <Select
              value={String(value.lr ?? preset.training.lr)}
              onChange={(v) => set('lr', parseFloat(v))}
              options={LR_OPTS.map(o => ({ value: String(o.value), label: o.label }))}
            />
          </FormRow>
          <FormRow label="LR scheduler" tip={TIPS.lr_scheduler}>
            <Select
              value={value.lr_scheduler ?? 'none'}
              onChange={(v) => set('lr_scheduler', v as 'none' | 'cosine' | 'plateau')}
              options={[
                { value: 'none',    label: 'None (constant)' },
                { value: 'cosine',  label: 'Cosine warm restarts' },
                { value: 'plateau', label: 'Reduce on plateau' },
              ]}
            />
          </FormRow>
          {(value.loss ?? preset.training.loss) === 'l1+lpips' && (
            <FormRow label="LPIPS λ" tip={TIPS.lambda_lpips} hint="0.1–0.3 recommended; >0.5 risks artifacts">
              <Input
                type="number" step="0.05" min={0} max={1}
                value={value.lambda_lpips ?? preset.training.lambda_lpips}
                onChange={(e) => set('lambda_lpips', parseFloat(e.target.value))}
              />
            </FormRow>
          )}
          {(value.loss ?? preset.training.loss) === 'weighted' && (
            <>
              <FormRow label="L1 weight" tip={TIPS.l1_weight}>
                <Input type="number" step="0.05" min={0} max={10}
                  value={value.l1_weight ?? preset.training.l1_weight ?? 0.5}
                  onChange={(e) => set('l1_weight', parseFloat(e.target.value))} />
              </FormRow>
              <FormRow label="L2 weight" tip={TIPS.l2_weight}>
                <Input type="number" step="0.05" min={0} max={10}
                  value={value.l2_weight ?? preset.training.l2_weight ?? 0.5}
                  onChange={(e) => set('l2_weight', parseFloat(e.target.value))} />
              </FormRow>
              <FormRow label="LPIPS weight" tip={TIPS.lpips_weight}>
                <Input type="number" step="0.05" min={0} max={10}
                  value={value.lpips_weight ?? preset.training.lpips_weight ?? 0.0}
                  onChange={(e) => set('lpips_weight', parseFloat(e.target.value))} />
              </FormRow>
            </>
          )}
          <FormRow label="Mixed precision (fp16)" tip={TIPS.use_amp}>
            <Toggle
              checked={value.use_amp ?? true}
              onChange={(c) => set('use_amp', c)}
              label="Enable AMP (faster on Tensor-core GPUs)"
            />
          </FormRow>
        </Grid2>
      </Section>

      <Section title="Schedule">
        <Grid2>
          <FormRow label="Batch size" tip={TIPS.batch_size} hint="0 = auto-detect by GPU memory">
            <Select
              value={String(value.batch_size ?? 0)}
              onChange={(v) => set('batch_size', parseInt(v, 10))}
              options={BATCH_SIZES.map(b => ({
                value: String(b),
                label: b === 0 ? 'Auto' : String(b),
              }))}
            />
          </FormRow>
          <FormRow label="Iterations per epoch" tip={TIPS.iterations_per_epoch}>
            <Input
              type="number" min={1} max={10000}
              value={value.iterations_per_epoch ?? preset.training.iterations_per_epoch}
              onChange={(e) => set('iterations_per_epoch', parseInt(e.target.value || '1', 10))}
            />
          </FormRow>
          <FormRow label="Max steps" tip={TIPS.max_steps} hint="0 = unlimited (run until you stop it)">
            <Input
              type="number" min={0}
              value={value.max_steps ?? 0}
              onChange={(e) => set('max_steps', parseInt(e.target.value || '0', 10))}
            />
          </FormRow>
          <FormRow label="Progressive resolution" tip={TIPS.progressive_resolution}>
            <Toggle
              checked={value.progressive_resolution ?? preset.progressive_resolution}
              onChange={(c) => set('progressive_resolution', c)}
              label="Quarter → half → full"
            />
          </FormRow>
          {/* Early stopping — folded into Schedule (it's part of when a run ends). */}
          <FormRow label="Plateau detection" tip={TIPS.es_enabled}>
            <Toggle
              checked={value.es_enabled ?? false}
              onChange={(c) => set('es_enabled', c)}
              label="Watch for validation plateau"
            />
          </FormRow>
          {value.es_enabled && (
            <>
              <FormRow label="Patience (epochs)" tip={TIPS.es_patience}>
                <Input
                  type="number" min={5} max={200}
                  value={value.es_patience ?? 30}
                  onChange={(e) => set('es_patience', parseInt(e.target.value || '30', 10))}
                />
              </FormRow>
              <FormRow label="Stop on plateau" tip={TIPS.es_stop}>
                <Toggle
                  checked={value.es_stop ?? false}
                  onChange={(c) => set('es_stop', c)}
                  label="Auto-stop training (else just notify)"
                />
              </FormRow>
            </>
          )}
        </Grid2>
      </Section>

      <Section title="Mask & Skip-empty">
        <Grid2>
          <FormRow label="Use mask loss" tip={TIPS.use_mask_loss}>
            <Toggle
              checked={value.use_mask_loss ?? preset.mask.use_mask_loss ?? false}
              onChange={(c) => set('use_mask_loss', c)}
              label="Weight loss by mask (white = important)"
            />
          </FormRow>
          {(value.use_mask_loss ?? preset.mask.use_mask_loss ?? false) && (
            <FormRow label="Mask weight" tip={TIPS.mask_weight}>
              <Input
                type="number" step="0.5" min={1} max={1000}
                value={value.mask_weight ?? preset.mask.mask_weight ?? 10.0}
                onChange={(e) => set('mask_weight', parseFloat(e.target.value || '10'))}
              />
            </FormRow>
          )}
          <FormRow label="Use mask as 4th input channel" tip={TIPS.use_mask_input}>
            <Toggle
              checked={value.use_mask_input ?? false}
              onChange={(c) => set('use_mask_input', c)}
              label="Feed mask into the model"
            />
          </FormRow>
          <FormRow label="Auto-mask" tip={TIPS.use_auto_mask}>
            <Toggle
              checked={value.use_auto_mask ?? preset.mask.use_auto_mask}
              onChange={(c) => set('use_auto_mask', c)}
              label="Auto-generate masks from |src − dst|"
            />
          </FormRow>
          {(value.use_auto_mask ?? preset.mask.use_auto_mask) && (
            <FormRow label="Auto-mask gamma" tip={TIPS.auto_mask_gamma} hint="<1 expands white regions; 0.5 for beauty">
              <div className="flex items-center gap-2">
                <Input
                  // step="any" lets the user type arbitrary decimals like 0.55
                  // — `step="0.1"` makes Chrome reject values that don't land on
                  // a multiple of the step from `min`.
                  type="number" step="any" min={0.1} max={5}
                  value={value.auto_mask_gamma ?? preset.mask.auto_mask_gamma ?? 1.0}
                  onChange={(e) => set('auto_mask_gamma', parseFloat(e.target.value || '1'))}
                  className="flex-1"
                />
                {onAutoMaskPreview && (
                  <button
                    type="button"
                    onClick={onAutoMaskPreview}
                    disabled={!autoMaskPreviewReady}
                    title={autoMaskPreviewReady
                      ? 'Visualize the auto-mask at this gamma on a few sample frames'
                      : 'Pick a project folder first'}
                    className="px-3 py-2 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff] disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    Preview Mask
                  </button>
                )}
              </div>
            </FormRow>
          )}
          <FormRow label="Skip empty patches" tip={TIPS.skip_empty_patches}>
            <Toggle
              checked={value.skip_empty_patches ?? preset.skip_empty_patches}
              onChange={(c) => set('skip_empty_patches', c)}
              label="Skip patches where src ≈ dst"
            />
          </FormRow>
          {(value.skip_empty_patches ?? preset.skip_empty_patches) && (
            <FormRow label="Skip threshold" tip={TIPS.skip_empty_threshold} hint="Mean |src−dst| below this counts as empty">
              <div className="flex items-center gap-2">
                <Input
                  // step="any" — same reason as auto_mask_gamma above.
                  // Chrome was rejecting 6.9 etc. because step=0.5 from min=0.1
                  // restricts valid values to 0.1, 0.6, 1.1, 1.6, ...
                  type="number" step="any" min={0.1} max={50}
                  value={value.skip_empty_threshold ?? 3.0}
                  onChange={(e) => set('skip_empty_threshold', parseFloat(e.target.value || '3'))}
                  className="flex-1"
                />
                {onPreviewFilter && (
                  <button
                    type="button"
                    onClick={onPreviewFilter}
                    disabled={!previewFilterReady}
                    title={previewFilterReady
                      ? 'Visualize which patches will be kept vs skipped at this threshold'
                      : 'Pick a project folder first'}
                    className="px-3 py-2 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff] disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    Preview Filter
                  </button>
                )}
              </div>
            </FormRow>
          )}
        </Grid2>
      </Section>

      <Section title="Augmentations">
        <AugmentationsEditor
          value={value.augs ?? {}}
          onChange={(augs) => set('augs', augs)}
        />
      </Section>

      <Section title="Logging & Saving">
        <Grid2>
          <FormRow label="Log interval (steps)" tip={TIPS.log_interval}>
            <Input
              type="number" min={1} max={1000}
              value={value.log_interval ?? 5}
              onChange={(e) => set('log_interval', parseInt(e.target.value || '5', 10))}
            />
          </FormRow>
          <FormRow label="Preview every N batches" tip={TIPS.preview_batch_interval} hint="0 = disabled">
            <Input
              type="number" min={0} max={1000}
              value={value.preview_batch_interval ?? 0}
              onChange={(e) => set('preview_batch_interval', parseInt(e.target.value || '0', 10))}
            />
          </FormRow>
          <FormRow label="Preview refresh rate (s)" tip={TIPS.preview_refresh_rate}>
            <Input
              type="number" min={0} max={1000}
              value={value.preview_refresh_rate ?? 5}
              onChange={(e) => set('preview_refresh_rate', parseInt(e.target.value || '5', 10))}
            />
          </FormRow>
          <FormRow label="Keep last N checkpoints" tip={TIPS.keep_last_checkpoints}>
            <Input
              type="number" min={1} max={50}
              value={value.keep_last_checkpoints ?? 4}
              onChange={(e) => set('keep_last_checkpoints', parseInt(e.target.value || '4', 10))}
            />
          </FormRow>
        </Grid2>
      </Section>

      <Section title="Auto-export">
        <div className="space-y-4">
          <FormRow label="Export interval (epochs)" tip={TIPS.auto_export_interval} hint="0 = disabled">
            <Input
              type="number" min={0} max={1000}
              value={value.auto_export_interval ?? 0}
              onChange={(e) => set('auto_export_interval', parseInt(e.target.value || '0', 10))}
            />
          </FormRow>
          {/* Flame and Nuke always pair side-by-side — they're the two export
              targets and reading one without the other is confusing. */}
          <div className="grid grid-cols-2 gap-4">
            <FormRow label="Flame / After Effects" tip={TIPS.auto_export_flame}>
              <Toggle
                checked={value.auto_export_flame ?? false}
                onChange={(c) => set('auto_export_flame', c)}
                label="Export ONNX"
              />
            </FormRow>
            <FormRow label="Nuke" tip={TIPS.auto_export_nuke}>
              <Toggle
                checked={value.auto_export_nuke ?? false}
                onChange={(c) => set('auto_export_nuke', c)}
                label="Export TorchScript"
              />
            </FormRow>
          </div>
        </div>
      </Section>
    </div>
  )
}

// ── Subcomponents ──────────────────────────────────────────────────────────

function Section({ title, children, defaultOpen = false }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-[#e5e7eb] rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-[#F9FAFB] hover:bg-[#F3F4F6] transition-colors text-left"
      >
        <span className="text-sm font-semibold text-[#374151]">{title}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 150ms' }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
      {open && <div className="px-4 py-4 bg-white">{children}</div>}
    </div>
  )
}

function Grid2({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{children}</div>
}

function Select({ value, onChange, options }: {
  value: string; onChange: (v: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full border border-[#e5e7eb] rounded-lg px-3.5 py-2.5 text-sm bg-white focus:outline-none focus:border-[#ae69f4] focus:ring-3 focus:ring-[#ae69f4]/10"
    >
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

function Toggle({ checked, onChange, label }: {
  checked: boolean; onChange: (c: boolean) => void; label: string
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-[#374151] cursor-pointer select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-[#ae69f4] w-4 h-4"
      />
      <span>{label}</span>
    </label>
  )
}

// ── Augmentations editor (4 toggles, each expands to its own controls) ─────

function AugmentationsEditor({ value, onChange }: {
  value: AugConfig; onChange: (v: AugConfig) => void
}) {
  const set = <K extends keyof AugConfig>(k: K, v: AugConfig[K]) =>
    onChange({ ...value, [k]: v })

  return (
    <div className="space-y-3">
      <AugRow
        label="Horizontal flip"
        tip={TIPS.aug_hflip}
        enabled={!!value.hflip}
        onToggle={(on) => set('hflip', on ? { p: 0.5 } : undefined)}
      >
        {value.hflip && (
          <Slider label="Probability"
            value={value.hflip.p} min={0} max={1} step={0.05}
            onChange={(p) => set('hflip', { p })}
          />
        )}
      </AugRow>

      <AugRow
        label="Random affine"
        tip={TIPS.aug_affine}
        enabled={!!value.affine}
        onToggle={(on) => set('affine', on ? {
          p: 0.3,
          scale_min: 0.9, scale_max: 1.1,
          translate_min: -0.1, translate_max: 0.1,
          rotate_min: -3, rotate_max: 3,
          shear_min: -1, shear_max: 1,
          keep_ratio: true,
        } : undefined)}
      >
        {value.affine && (
          <div className="space-y-2 text-xs">
            <Slider label="Probability" value={value.affine.p} min={0} max={1} step={0.05}
              onChange={(p) => set('affine', { ...value.affine!, p })} />
            <Range label="Scale" min={value.affine.scale_min} max={value.affine.scale_max} step={0.05}
              onMin={(v) => set('affine', { ...value.affine!, scale_min: v })}
              onMax={(v) => set('affine', { ...value.affine!, scale_max: v })} />
            <Range label="Rotate (°)" min={value.affine.rotate_min} max={value.affine.rotate_max} step={1}
              onMin={(v) => set('affine', { ...value.affine!, rotate_min: v })}
              onMax={(v) => set('affine', { ...value.affine!, rotate_max: v })} />
            <Toggle checked={value.affine.keep_ratio}
              onChange={(c) => set('affine', { ...value.affine!, keep_ratio: c })}
              label="Keep aspect ratio (paired-image safe)" />
          </div>
        )}
      </AugRow>

      <AugRow
        label="Random gamma"
        tip={TIPS.aug_gamma}
        enabled={!!value.gamma}
        onToggle={(on) => set('gamma', on ? { p: 0.3, gamma_min: 80, gamma_max: 120 } : undefined)}
      >
        {value.gamma && (
          <div className="space-y-2 text-xs">
            <Slider label="Probability" value={value.gamma.p} min={0} max={1} step={0.05}
              onChange={(p) => set('gamma', { ...value.gamma!, p })} />
            <Range label="Gamma %" min={value.gamma.gamma_min} max={value.gamma.gamma_max} step={5}
              onMin={(v) => set('gamma', { ...value.gamma!, gamma_min: v })}
              onMax={(v) => set('gamma', { ...value.gamma!, gamma_max: v })} />
          </div>
        )}
      </AugRow>

      <AugRow
        label="Color (brightness/contrast/saturation)"
        tip={TIPS.aug_color}
        enabled={!!value.color}
        onToggle={(on) => set('color', on ? {
          p: 0.3,
          brightness_min: -0.1, brightness_max: 0.1,
          contrast_min: -0.1, contrast_max: 0.1,
          saturation_min: -10, saturation_max: 10,
        } : undefined)}
      >
        {value.color && (
          <div className="space-y-2 text-xs">
            <Slider label="Probability" value={value.color.p} min={0} max={1} step={0.05}
              onChange={(p) => set('color', { ...value.color!, p })} />
            <Range label="Brightness" min={value.color.brightness_min} max={value.color.brightness_max} step={0.05}
              onMin={(v) => set('color', { ...value.color!, brightness_min: v })}
              onMax={(v) => set('color', { ...value.color!, brightness_max: v })} />
            <Range label="Contrast" min={value.color.contrast_min} max={value.color.contrast_max} step={0.05}
              onMin={(v) => set('color', { ...value.color!, contrast_min: v })}
              onMax={(v) => set('color', { ...value.color!, contrast_max: v })} />
            <Range label="Saturation" min={value.color.saturation_min} max={value.color.saturation_max} step={1}
              onMin={(v) => set('color', { ...value.color!, saturation_min: v })}
              onMax={(v) => set('color', { ...value.color!, saturation_max: v })} />
          </div>
        )}
      </AugRow>
    </div>
  )
}

function AugRow({ label, tip, enabled, onToggle, children }: {
  label: string
  tip?: string
  enabled: boolean
  onToggle: (on: boolean) => void
  children?: React.ReactNode
}) {
  return (
    <div className={`border rounded-md ${enabled ? 'border-[#e9d5ff] bg-[#FCFAFE]' : 'border-[#e5e7eb] bg-white'}`}>
      <label className="flex items-center gap-2 px-3 py-2.5 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
          className="accent-[#ae69f4] w-4 h-4"
        />
        <span className="text-sm font-medium text-[#374151]">{label}</span>
        {tip && <InfoTip text={tip} />}
      </label>
      {enabled && children && (
        <div className="px-3 pb-3 pt-1 border-t border-[#F3E8FF]">
          {children}
        </div>
      )}
    </div>
  )
}

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-[#6b7280] w-24 flex-shrink-0">{label}</span>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-[#ae69f4]"
      />
      <span className="text-xs font-mono text-[#374151] w-12 text-right">{value.toFixed(2)}</span>
    </div>
  )
}

function Range({ label, min, max, step, onMin, onMax }: {
  label: string; min: number; max: number; step: number
  onMin: (v: number) => void; onMax: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-[#6b7280] w-24 flex-shrink-0">{label}</span>
      <Input type="number" step={step} value={min}
        onChange={(e) => onMin(parseFloat(e.target.value))}
        className="!py-1 !px-2 text-xs w-20" />
      <span className="text-xs text-[#9ca3af]">to</span>
      <Input type="number" step={step} value={max}
        onChange={(e) => onMax(parseFloat(e.target.value))}
        className="!py-1 !px-2 text-xs w-20" />
    </div>
  )
}
