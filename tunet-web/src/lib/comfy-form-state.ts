/**
 * Serialized form state for an EZ-Comfy job submission.
 *
 * Stashed in `env.TUNET_FORM_STATE` on every comfy submit (JSON.stringified) so
 * the job detail page can show the prompt + knobs the render actually used, and
 * the Clone button can rehydrate the form. Mirrors the training-side
 * `spark-form-state.ts`, but with the comfy shape (preset key + the raw param
 * `values` map the form posts).
 *
 * Discriminated from a training stash by the required `presetKey` field: a
 * training stash carries `preset`/`gpuKey` instead, so `parseComfyFormState`
 * returns null on it (and vice-versa) — the two never cross-contaminate even
 * though they share the env key.
 *
 * The input clip is intentionally NOT stashed — it's re-picked on clone, the
 * same way the training clone doesn't re-use the dataset folder.
 *
 * Client-safe — no `server-only` imports (read by the client comfy form).
 */

export const COMFY_FORM_STATE_VERSION = 1

export type ComfyComputeMode = 'instant' | 'smart'

export interface ComfyFormState {
  version:   number
  presetKey: string
  gpu:       string
  mode:      ComfyComputeMode
  /** Job name the user typed (may be ''). */
  name:      string
  /** The preset param values the form posted (prompt, steps, fps, …), minus
      the structural file inputs (video/mask/face) and the __name marker. */
  values:    Record<string, unknown>
}

export interface ComfyFormStateInput {
  presetKey: string
  gpu:       string
  mode:      ComfyComputeMode
  name:      string
  values:    Record<string, unknown>
}

export function buildComfyFormState(input: ComfyFormStateInput): ComfyFormState {
  return { version: COMFY_FORM_STATE_VERSION, ...input }
}

/**
 * Parse a `TUNET_FORM_STATE` env value back into a ComfyFormState. Tolerant of
 * missing/extra fields; anything unparseable (or a training-shaped stash)
 * returns null so the caller can fall back to `env.TUNET_PRESET`.
 */
export function parseComfyFormState(raw: string | undefined | null): ComfyFormState | null {
  if (!raw) return null
  try {
    const p = JSON.parse(raw) as Partial<ComfyFormState>
    if (!p || typeof p !== 'object') return null
    if (typeof p.presetKey !== 'string') return null   // not a comfy stash
    return {
      version:   typeof p.version === 'number' ? p.version : 0,
      presetKey: p.presetKey,
      gpu:       typeof p.gpu === 'string' ? p.gpu : 'rtxpro6000',
      mode:      p.mode === 'smart' ? 'smart' : 'instant',
      name:      typeof p.name === 'string' ? p.name : '',
      values:    p.values && typeof p.values === 'object' ? p.values as Record<string, unknown> : {},
    }
  } catch {
    return null
  }
}
