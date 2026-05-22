/**
 * Server-side helpers for EZ-Comfy — a web port of comfy_spark/comfy_launch.py.
 *
 * A "comfy job" is just another Spark job: pack a tiny tarball
 * (comfy_run.py + workflow.json + patches.json + the input clip) that extracts
 * to /input/, then submit it with the preset's ComfyUI image and the command
 * `bash -c "<python> /input/comfy_run.py"`. comfy_run.py (shipped verbatim from
 * the repo) does the heavy lifting on the Spark node: clone node packs, fetch
 * weights, convert the UI graph to API format, apply patches, render, and
 * self-upload the outputs to ShareSync.
 *
 * This module mirrors comfy_launch.py's CLIENT side (build patches, build env,
 * pack, pick instance) and reuses spark.ts (submitJob / uploadInputTarball /
 * getToken) so it inherits the per-user Keycloak auth + the streaming uploader.
 */

import 'server-only'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as os from 'node:os'
import * as crypto from 'node:crypto'
import * as tar from 'tar'
import { pipeline } from 'node:stream/promises'
import { findTunetRepoRoot } from './spark-packer'
import { GPU_TYPES, type GpuKey } from './spark'

// ── Preset model ──────────────────────────────────────────────────────────────

export interface ComfyParamSpec {
  node:    string
  path:    string
  default?: unknown
  ui?:     Record<string, unknown>
}

/** A canonical-docs link on a preset (or a bare URL string — both accepted). */
export interface ComfyDocLink { label?: string; url: string }

/** Plain-language, jargon-free blurb shown in the UI (CLI --list-presets / web About card). */
export interface ComfyAbout { what?: string; inputs?: string; key_knobs?: string }

export interface ComfyPreset {
  key:           string
  order?:        number          // dropdown sort key (lower first); default 100
  description?:   string
  docs?:         ComfyDocLink[] | string[] | string   // link(s) to canonical docs
  tags?:         string[]                              // for sifting/filtering
  about?:        ComfyAbout                            // plain-language description
  base?:         string
  image?:        string
  comfy_home?:   string
  comfy_bundle?: string
  python?:       string
  gpu?:          string
  mode?:         string
  extra_args?:   string
  workflow:      string                 // filename under presets/
  node_packs?:   string[]
  models?:       { url: string; dest: string }[]
  lora_default?: string
  lora_name?:    string
  lora_chain?:   { anchor: string; slot: number }
  lora_catalog?: string
  output_prefix?: { nodes: string[]; path?: string; template?: string }
  ui?:           Record<string, unknown>
  params:        Record<string, ComfyParamSpec>
}

/** comfy_spark/ sits at the repo root next to train.py. */
export function comfyDir(): string {
  return path.join(findTunetRepoRoot(), 'comfy_spark')
}

const KEY_RE = /^[A-Za-z0-9._-]+$/

export function loadComfyPresets(): ComfyPreset[] {
  const dir = path.join(comfyDir(), 'presets')
  if (!fs.existsSync(dir)) return []
  const out: ComfyPreset[] = []
  for (const f of fs.readdirSync(dir).sort()) {
    if (!f.endsWith('.preset.json')) continue
    try {
      const p = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'))
      out.push({ ...p, key: f.slice(0, -'.preset.json'.length) })
    } catch { /* skip a malformed preset rather than break the whole list */ }
  }
  // Explicit `order` floats presets to the top (e.g. ObscuraRemova); ties keep
  // the filename order. Stable sort (V8) preserves it.
  out.sort((a, b) => (a.order ?? 100) - (b.order ?? 100))
  return out
}

export function loadComfyPreset(key: string): ComfyPreset | null {
  if (!KEY_RE.test(key)) return null
  const file = path.join(comfyDir(), 'presets', `${key}.preset.json`)
  if (!fs.existsSync(file)) return null
  try {
    return { ...JSON.parse(fs.readFileSync(file, 'utf8')), key }
  } catch {
    return null
  }
}

/**
 * The param name a preset's SECONDARY uploaded input maps to, or null. A face-swap
 * preset declares it via `ui.secondary_input.param` (e.g. "face"); a VACE-style
 * preset that takes a `mask` param uses that. Two-input presets only.
 */
export function comfySecondaryParam(preset: ComfyPreset): string | null {
  const sec = preset.ui?.secondary_input as { param?: string } | undefined
  if (sec) return sec.param ?? 'face'   // keep in sync with secondaryOf() in comfy-form.tsx
  if (preset.params?.mask) return 'mask'
  return null
}

/** The workflow JSON a preset points at (UI-format graph for our presets). */
export function loadComfyWorkflow(preset: ComfyPreset): unknown {
  const file = path.join(comfyDir(), 'presets', preset.workflow)
  if (!fs.existsSync(file)) throw new Error(`workflow ${preset.workflow} not found for preset ${preset.key}`)
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

/** A LoRA catalog (name → {url,file,triggers,...}) referenced by a preset. */
export function loadLoraCatalog(preset: ComfyPreset): Record<string, { url: string; file: string }> {
  const name = preset.lora_catalog
  if (!name || !KEY_RE.test(name.replace(/\.json$/, ''))) return {}
  const file = path.join(comfyDir(), 'loras', name)
  if (!fs.existsSync(file)) return {}
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8')).loras ?? {}
  } catch {
    return {}
  }
}

// ── Patches ─────────────────────────────────────────────────────────────────

export interface ComfyPatch { node: string; path: string; value: unknown }

/**
 * Build the patch list from preset params + the form values, mirroring
 * comfy_launch.py: each declared param maps to a node.path; input params get
 * their uploaded file's basename; output_prefix nodes get a sanitized prefix.
 *
 * `inputBasename` fills the `video` param (the primary clip). `secondary` fills a
 * second input param — the face image (`face`) for ltx_faceswap or the mask for
 * wan_vace_inpaint — mirroring comfy_launch.py's `--face` / `--mask`.
 */
export function buildComfyPatches(
  preset: ComfyPreset,
  values: Record<string, unknown>,
  inputBasename: string | null,
  secondary?: { param: string; basename: string } | null,
): ComfyPatch[] {
  const inputNames: Record<string, string> = {}
  if (inputBasename) inputNames['video'] = inputBasename
  if (secondary?.basename) inputNames[secondary.param] = secondary.basename

  const ops: ComfyPatch[] = []
  for (const [name, spec] of Object.entries(preset.params ?? {})) {
    let val: unknown = values[name]
    if (val === undefined || val === null || val === '') {
      val = inputNames[name] ?? spec.default
    }
    if (val === undefined || val === null) continue
    ops.push({ node: String(spec.node), path: spec.path, value: val })
  }

  const op = preset.output_prefix
  if (op && inputBasename) {
    const stem  = inputBasename.replace(/\.[^.]+$/, '')
    const model = String(preset.base ?? '').split('-')[0]
    const raw   = (op.template ?? '{stem}_{preset}')
      .replaceAll('{stem}', stem)
      .replaceAll('{preset}', preset.key)
      .replaceAll('{model}', model)
    const prefix = (raw.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 120)) || 'output'
    for (const nid of op.nodes ?? []) {
      ops.push({ node: String(nid), path: op.path ?? 'inputs.filename_prefix', value: prefix })
    }
  }
  return ops
}

// ── Env / instance / command ──────────────────────────────────────────────────

export interface ComfyLora { url: string; file: string; strength: number }

/**
 * The job env block comfy_run.py reads. `uploadToken` is the Spark bearer it
 * uses post-render to self-upload /output to ShareSync (the agent doesn't sync
 * it). The render can outlast that short-lived token, so we also pass refresh
 * material (TUNET_REFRESH_TOKEN/URL): comfy_run mints a fresh bearer via the
 * /api/spark/refresh proxy right before uploading. uploadToken stays as the
 * initial/fallback bearer for jobs without refresh material.
 */
export function buildComfyEnv(
  preset: ComfyPreset,
  uploadToken: string,
  opts: { hasPatches: boolean; loras?: ComfyLora[]; readyTimeout?: number; refresh?: { refreshToken: string; refreshUrl: string } | null },
): Record<string, string> {
  const env: Record<string, string> = {
    COMFY_WORKFLOW:       '/input/workflow.json',
    COMFY_READY_TIMEOUT:  String(opts.readyTimeout ?? 300),
    COMFY_RUN_ID:         crypto.randomBytes(16).toString('hex'),
    COMFY_UPLOAD_TOKEN:   uploadToken,
  }
  if (opts.refresh) {
    env.TUNET_REFRESH_TOKEN = opts.refresh.refreshToken
    env.TUNET_REFRESH_URL   = opts.refresh.refreshUrl
  }
  if (opts.hasPatches) env.COMFY_PATCHES = '/input/patches.json'
  if (preset.lora_default) {
    env.COMFY_LORA_URL  = preset.lora_default
    env.COMFY_LORA_NAME = preset.lora_name ?? 'lora.safetensors'
  }
  if (opts.loras?.length && preset.lora_chain) {
    env.COMFY_LORAS      = JSON.stringify(opts.loras.map(l => ({ url: l.url, file: l.file, strength: l.strength })))
    env.COMFY_LORA_CHAIN = JSON.stringify(preset.lora_chain)
  }
  if (preset.extra_args)   env.COMFY_EXTRA_ARGS  = preset.extra_args
  if (preset.comfy_home)   env.COMFY_HOME        = preset.comfy_home
  if (preset.comfy_bundle) env.COMFY_BUNDLE      = preset.comfy_bundle
  if (preset.node_packs)   env.COMFY_FETCH_NODES = JSON.stringify(preset.node_packs)
  if (preset.models)       env.COMFY_FETCH_MODELS = JSON.stringify(preset.models)
  return env
}

/** Resolve the Spark SKU from the preset's gpu (or an override). */
export function comfyInstanceType(preset: ComfyPreset, gpuOverride?: string): string {
  const key = (gpuOverride || preset.gpu || 'rtxpro6000') as GpuKey
  return GPU_TYPES[key]?.sku ?? GPU_TYPES.rtxpro6000.sku
}

/** Run under the image's real interpreter (e.g. python3.13 on the yanwk image). */
export function comfyCommand(preset: ComfyPreset): string[] {
  return ['bash', '-c', `${preset.python ?? 'python3'} /input/comfy_run.py`]
}

// ── Pack ───────────────────────────────────────────────────────────────────────

/**
 * Stream a comfy job tarball to disk: workflow.json + patches.json +
 * comfy_run.py + the input media (by basename). Never holds a clip in RAM —
 * everything is written/copied on disk and tar-streamed (the same discipline as
 * spark-packer, so a multi-hundred-MB clip won't OOM the box). `inputs` is the
 * list of uploaded media (the primary clip, plus a secondary face/mask for
 * two-input presets); each is packed by bare basename, which is how the workflow
 * references it. Caller deletes the returned tarball after upload.
 */
export async function packComfyTarball(opts: {
  workflow: unknown
  patches:  ComfyPatch[] | null
  inputs:   { path: string; basename: string }[]
}): Promise<{ tarballPath: string; fileCount: number; compressedSize: number }> {
  const runner = path.join(comfyDir(), 'comfy_run.py')
  if (!fs.existsSync(runner)) throw new Error(`comfy_run.py not found at ${runner}`)

  const stamp = Date.now().toString(36)
  const stage = path.join(os.tmpdir(), `comfy-pack-${stamp}-${process.pid}`)
  await fs.promises.mkdir(stage, { recursive: true })
  let fileCount = 0
  try {
    await fs.promises.writeFile(path.join(stage, 'workflow.json'), JSON.stringify(opts.workflow))
    fileCount += 1
    if (opts.patches) {
      await fs.promises.writeFile(path.join(stage, 'patches.json'), JSON.stringify(opts.patches))
      fileCount += 1
    }
    await fs.promises.copyFile(runner, path.join(stage, 'comfy_run.py'))
    fileCount += 1
    for (const inp of opts.inputs) {
      // basename only — the workflow references each file by bare filename
      await fs.promises.copyFile(inp.path, path.join(stage, path.basename(inp.basename)))
      fileCount += 1
    }

    const entries     = await fs.promises.readdir(stage)
    const tarballPath  = path.join(os.tmpdir(), `comfy-tarball-${stamp}-${process.pid}.tar.gz`)
    await pipeline(
      tar.create({ gzip: { level: 6 }, cwd: stage, follow: false, portable: true }, entries),
      fs.createWriteStream(tarballPath),
    )
    const compressedSize = (await fs.promises.stat(tarballPath)).size
    return { tarballPath, fileCount, compressedSize }
  } finally {
    await fs.promises.rm(stage, { recursive: true, force: true }).catch(() => {})
  }
}
