/**
 * Serialized form state for a Spark training job submission.
 *
 * Stashed in `env.TUNET_FORM_STATE` on every submit (JSON.stringified) so we
 * can fully rehydrate the new-job form when the user wants to clone a prior
 * job — Spark's job record alone doesn't carry enough detail (no name,
 * truncated argv parsing, no user-friendly preset/advanced split).
 *
 * Bumped versions: if the schema changes, increment `version` and add an
 * adapter in `parseFormState()` so old jobs still clone cleanly.
 *
 * Client-safe — no `server-only` imports.
 */

import type { AdvancedOverrides, PresetKey } from './spark-presets'

export const FORM_STATE_VERSION = 3

export type TrainingMode = 'new' | 'resume' | 'finetune'
export type ComputeMode  = 'instant' | 'smart'

export interface SourceJobRef {
  /**
   * Spark job id of the prior run we're resuming from / fine-tuning off of.
   * For local-checkpoint uploads (where the user trained off-Spark), this
   * is the sentinel string 'local-upload' and `localCheckpointStageId`
   * carries the upload-stage handle the server reads the .pth from.
   */
  jobId: string
  /** Friendly name at clone time — display-only, for the "Resuming X" badge. */
  jobLabel: string
  /**
   * Filename of the chosen .pth inside the source job's output dir
   * (relative, no path). Resume mode picks `*_tunet_latest.pth` automatically;
   * fine-tune mode lets the user pick any .pth.
   */
  checkpointName: string
  /**
   * Set when the .pth came from a local browser upload rather than a prior
   * Spark job. Value is the stageId returned from /api/spark/upload-stage,
   * which the training-jobs route uses to read the file from disk instead of
   * fetching it from Spark's ShareSync. `jobId` is 'local-upload' in this case.
   */
  localCheckpointStageId?: string
}

export interface SerializedFormState {
  version:           number
  /** Defaults to 'new' for older form-state stashes (version 1). */
  mode:              TrainingMode
  source?:           SourceJobRef
  preset:            PresetKey
  gpuKey:            string
  /** Defaults to 'instant' for stashes older than v3. */
  computeMode:       ComputeMode
  /** Smart-mode retry budget. Defaults to 1 for older stashes / instant mode. */
  maxRetriesOnInterrupt: number
  advanced:          AdvancedOverrides
  maxSteps:          number
  idleHoldSeconds:   number
  pairs:             number
  skipThreshold:     number
  alerts: {
    email:           string
    plateau:         boolean
    diverging:       boolean
  }
  /**
   * Manual ShareSync paths (only meaningful when no folder was picked).
   * On clone we always show the folder picker again — these are kept just
   * so the "Manual Paths" toggle reopens with the same values.
   */
  manual?: {
    srcDir:    string
    dstDir:    string
    valSrcDir: string
    valDstDir: string
  }
}

export interface FormStateBuildInput {
  mode:            TrainingMode
  source?:         SourceJobRef
  preset:          PresetKey
  gpuKey:          string
  computeMode:     ComputeMode
  maxRetriesOnInterrupt: number
  advanced:        AdvancedOverrides
  maxSteps:        number
  idleHoldSeconds: number
  pairs:           number
  skipThreshold:   number
  alerts: {
    email:     string
    plateau:   boolean
    diverging: boolean
  }
  manual?: SerializedFormState['manual']
}

export function buildFormState(input: FormStateBuildInput): SerializedFormState {
  return { version: FORM_STATE_VERSION, ...input }
}

/**
 * Parse a `TUNET_FORM_STATE` env value back into a SerializedFormState. Tolerant
 * of missing/extra fields — anything unparseable returns null and the caller
 * should fall back to inferring state from `command` argv + other env vars.
 */
export function parseFormState(raw: string | undefined | null): SerializedFormState | null {
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Partial<SerializedFormState>
    if (!parsed || typeof parsed !== 'object') return null
    if (typeof parsed.preset !== 'string')   return null
    if (typeof parsed.gpuKey !== 'string')   return null
    // Older form states (v1) didn't carry a mode — they were always 'new'.
    const mode: TrainingMode =
      parsed.mode === 'resume' || parsed.mode === 'finetune' ? parsed.mode : 'new'

    // Source ref is only meaningful for resume/finetune; on clone we *don't*
    // re-thread it into the new run automatically (the user may want a
    // different action) — the field is preserved so the badge can show
    // "this clone was originally a resume of <X>".
    const source: SourceJobRef | undefined =
      parsed.source && typeof parsed.source.jobId === 'string'
        ? {
            jobId:          parsed.source.jobId,
            jobLabel:       parsed.source.jobLabel ?? parsed.source.jobId,
            checkpointName: parsed.source.checkpointName ?? '',
          }
        : undefined

    // v3 added computeMode + maxRetriesOnInterrupt. Older stashes get the
    // back-compat defaults: instant mode (which is what every pre-v3 job
    // was) and a retry budget of 1.
    const computeMode: ComputeMode =
      parsed.computeMode === 'smart' ? 'smart' : 'instant'
    const maxRetriesOnInterrupt =
      typeof parsed.maxRetriesOnInterrupt === 'number'
        ? Math.max(0, Math.min(5, Math.floor(parsed.maxRetriesOnInterrupt)))
        : 1

    return {
      version:         typeof parsed.version === 'number' ? parsed.version : 0,
      mode,
      ...(source ? { source } : {}),
      preset:          parsed.preset as PresetKey,
      gpuKey:          parsed.gpuKey,
      computeMode,
      maxRetriesOnInterrupt,
      advanced:        (parsed.advanced ?? {}) as AdvancedOverrides,
      maxSteps:        typeof parsed.maxSteps        === 'number' ? parsed.maxSteps        : 0,
      idleHoldSeconds: typeof parsed.idleHoldSeconds === 'number' ? parsed.idleHoldSeconds : 0,
      pairs:           typeof parsed.pairs           === 'number' ? parsed.pairs           : 0,
      skipThreshold:   typeof parsed.skipThreshold   === 'number' ? parsed.skipThreshold   : 3.0,
      alerts: {
        email:     parsed.alerts?.email     ?? '',
        plateau:   parsed.alerts?.plateau   ?? true,
        diverging: parsed.alerts?.diverging ?? true,
      },
      ...(parsed.manual ? { manual: parsed.manual } : {}),
    }
  } catch {
    return null
  }
}
