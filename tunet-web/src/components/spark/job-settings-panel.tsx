/**
 * Renders the form state a job was submitted with — preset, GPU, compute mode,
 * batch/steps/resolution, advanced overrides, training alerts, etc.
 *
 * Reads `TUNET_FORM_STATE` from the job's env (serialized at submit time by
 * /api/spark/training-jobs) and parses it with the shared parseFormState
 * helper from spark-form-state.ts. Falls back to the loose TUNET_* env vars
 * (TUNET_PRESET, TUNET_GPU, etc.) for jobs that pre-date the form-state stash.
 *
 * Server-side rendered (no client interactivity needed — settings are
 * immutable once a job is submitted).
 */

import { parseFormState } from '@/lib/spark-form-state'
import type { SparkJob } from '@/lib/spark-types'
import { PRESETS } from '@/lib/spark-presets'

export function JobSettingsPanel({ job }: { job: SparkJob }) {
  const env = job.env ?? {}
  const stashed = parseFormState(env.TUNET_FORM_STATE)
  const advanced = stashed?.advanced ?? {}

  // Loose env fallbacks for jobs without a form-state stash
  const presetKey = stashed?.preset ?? env.TUNET_PRESET
  const gpuKey = stashed?.gpuKey ?? env.TUNET_GPU
  const presetName = presetKey && presetKey in PRESETS ? PRESETS[presetKey as keyof typeof PRESETS].name : presetKey

  // Compute mode + retry budget (smart only)
  const computeMode = stashed?.computeMode ?? (env.TUNET_COMPUTE_MODE === 'smart' ? 'smart' : 'instant')
  const retries = stashed?.maxRetriesOnInterrupt
  const computeLabel = computeMode === 'smart'
    ? `SmartCompute${retries !== undefined ? ` · retry budget ${retries}` : ''}`
    : 'InstantCompute (warm-pool)'

  // Training mode + source (resume / finetune)
  const trainingMode = stashed?.mode ?? env.TUNET_MODE ?? 'new'
  const sourceJobId = stashed?.source?.jobId ?? env.TUNET_SOURCE_JOB_ID
  const sourceCkpt = stashed?.source?.checkpointName ?? env.TUNET_SOURCE_CHECKPOINT

  // Alerts
  const alertEmail = stashed?.alerts.email ?? env.TUNET_ALERT_EMAIL ?? ''
  const alertsOn = (alertEmail || '').trim() !== ''
  const alertPlateau = stashed?.alerts.plateau ?? (env.TUNET_ALERT_PLATEAU === '1')
  const alertDiverging = stashed?.alerts.diverging ?? (env.TUNET_ALERT_DIVERGING === '1')

  // Resolution + model size — prefer overrides, fall back to preset defaults
  const presetObj = presetKey && presetKey in PRESETS ? PRESETS[presetKey as keyof typeof PRESETS] : null
  const resolution = advanced.resolution ?? presetObj?.data.resolution
  const modelSize = advanced.model_size_dims ?? presetObj?.model.model_size_dims
  const modelType = (advanced.model_type ?? presetObj?.model.model_type)?.toString().toUpperCase()
  const lossFn = advanced.loss ?? presetObj?.training.loss

  const batchSize = advanced.batch_size
  const batchLabel = batchSize === 0 || batchSize === undefined ? 'Auto (detect by GPU)' : String(batchSize)

  const maxSteps = stashed?.maxSteps
  const stepsLabel = maxSteps === undefined ? '—' : maxSteps === 0 ? 'Unlimited' : maxSteps.toLocaleString()

  const idleHold = stashed?.idleHoldSeconds
  const pairs = stashed?.pairs

  // Skip-empty + auto-mask
  const skipThreshold = stashed?.skipThreshold
  const useAutoMask = advanced.use_auto_mask
  const autoMaskGamma = advanced.auto_mask_gamma
  const maskWeight = advanced.mask_weight
  const useMaskLoss = advanced.use_mask_loss

  // If we have literally no form info, don't render the panel at all
  const hasAnything = stashed || presetKey || gpuKey || env.TUNET_JOB_NAME
  if (!hasAnything) return null

  return (
    <section>
      {/* Collapsed by default — users usually only need to glance at
          settings when something looks off; the chart / preview / log
          deserve the page-fold space otherwise. Click to expand. */}
      <details className="bg-white border border-[#e5e7eb] rounded-lg">
        <summary className="cursor-pointer select-none px-4 py-3 text-sm font-semibold text-[#374151] hover:bg-[#F9FAFB] rounded-lg flex items-center gap-2">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
               className="transition-transform details-marker-rotate">
            <polyline points="9 18 15 12 9 6" />
          </svg>
          Submitted settings
          {stashed && (
            <span className="ml-auto text-[10px] uppercase tracking-wider text-[#9ca3af] font-normal">
              form state v{stashed.version}
            </span>
          )}
        </summary>

        <div className="px-4 pb-4 pt-1 space-y-4 text-sm">
          {/* Top row: compute + training mode */}
          <Group title="Compute">
            <Row label="Mode" value={computeLabel} />
            <Row
              label="Training"
              value={
                trainingMode === 'new' ? 'New (train from scratch)' :
                trainingMode === 'resume' ? `Resume from ${sourceCkpt ?? '—'}${sourceJobId ? ` (job ${sourceJobId.slice(0, 8)}…)` : ''}` :
                trainingMode === 'finetune' ? `Fine-tune from ${sourceCkpt ?? '—'}${sourceJobId ? ` (job ${sourceJobId.slice(0, 8)}…)` : ''}` :
                trainingMode
              }
            />
            {gpuKey && <Row label="GPU" value={gpuKey.toUpperCase()} mono />}
          </Group>

          {/* Preset + model */}
          {(presetName || modelType || modelSize) && (
            <Group title="Preset & Model">
              {presetName && <Row label="Preset" value={presetName} />}
              {modelType && modelSize !== undefined && (
                <Row label="Model" value={`${modelType} ${modelSize}`} mono />
              )}
              {resolution !== undefined && <Row label="Resolution" value={`${resolution}px`} mono />}
            </Group>
          )}

          {/* Schedule */}
          {(maxSteps !== undefined || batchSize !== undefined || idleHold !== undefined || pairs !== undefined) && (
            <Group title="Schedule">
              <Row label="Batch size" value={batchLabel} mono />
              {maxSteps !== undefined && <Row label="Max steps" value={stepsLabel} mono />}
              {idleHold !== undefined && <Row label="Idle hold" value={`${idleHold}s`} mono />}
              {pairs !== undefined && pairs > 0 && <Row label="Frame pairs" value={String(pairs)} mono />}
            </Group>
          )}

          {/* Optimization */}
          {(lossFn || advanced.lr !== undefined || advanced.lambda_lpips !== undefined) && (
            <Group title="Optimization">
              {lossFn && <Row label="Loss" value={String(lossFn)} mono />}
              {advanced.lr !== undefined && <Row label="Learning rate" value={advanced.lr.toExponential(1)} mono />}
              {advanced.lambda_lpips !== undefined && (
                <Row label="LPIPS λ" value={String(advanced.lambda_lpips)} mono />
              )}
            </Group>
          )}

          {/* Mask + Skip */}
          {(useAutoMask || useMaskLoss || skipThreshold !== undefined) && (
            <Group title="Mask & Skip-empty">
              {useAutoMask !== undefined && (
                <Row label="Auto-mask" value={useAutoMask ? `enabled${autoMaskGamma !== undefined ? ` (γ=${autoMaskGamma})` : ''}` : 'off'} />
              )}
              {useMaskLoss && maskWeight !== undefined && (
                <Row label="Mask weight" value={String(maskWeight)} mono />
              )}
              {skipThreshold !== undefined && (
                <Row label="Skip-empty threshold" value={String(skipThreshold)} mono />
              )}
            </Group>
          )}

          {/* Alerts */}
          {alertsOn && (
            <Group title="Email alerts">
              <Row label="Recipient" value={alertEmail} mono small />
              <Row label="On plateau" value={alertPlateau ? 'yes' : 'no'} />
              <Row label="On diverging" value={alertDiverging ? 'yes' : 'no'} />
            </Group>
          )}

          {/* Raw env fallback for advanced users */}
          <details className="text-xs">
            <summary className="cursor-pointer text-[#9ca3af] hover:text-[#374151] select-none">
              Raw env vars ({Object.keys(env).filter(k => k.startsWith('TUNET_')).length})
            </summary>
            <div className="mt-2 bg-[#F9FAFB] border border-[#e5e7eb] rounded p-2 font-mono text-[10px] text-[#374151] overflow-x-auto">
              {Object.entries(env)
                .filter(([k]) => k.startsWith('TUNET_'))
                .map(([k, v]) => (
                  <div key={k} className="whitespace-pre-wrap break-all">
                    <span className="text-[#7E3AF2]">{k}</span>
                    <span className="text-[#9ca3af]">=</span>
                    <span>{k === 'TUNET_FORM_STATE' ? `<${(v ?? '').length} chars JSON>` : v}</span>
                  </div>
                ))}
            </div>
          </details>
        </div>
      </details>
    </section>
  )
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold mb-1.5">{title}</p>
      <div className="bg-[#F9FAFB] border border-[#e5e7eb] rounded-md divide-y divide-[#e5e7eb]">
        {children}
      </div>
    </div>
  )
}

function Row({ label, value, mono, small }: { label: string; value: string; mono?: boolean; small?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-1.5">
      <span className="text-xs text-[#6b7280]">{label}</span>
      <span className={`text-right text-[#111827] ${mono ? 'font-mono' : ''} ${small ? 'text-xs' : 'text-sm'} break-all`}>
        {value}
      </span>
    </div>
  )
}
