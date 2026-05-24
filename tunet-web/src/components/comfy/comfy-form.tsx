'use client'

// EZ-Comfy form — web port of the tkinter comfy_ui.py. Preset-driven: the form
// is rendered entirely from each preset's `ui` metadata (fetched from
// /api/comfy/presets), the input clip is chunk-uploaded, and submit streams
// pack/submit/upload phases from /api/comfy/submit. Once the Spark job is in,
// we hand off to the job detail page for live render logs + output download.
// (Extra stacked LoRAs beyond a preset's built-in one are a follow-up.)
//
// Shared by both /comfy (real app) and /comfy. The only thing that differs
// is where the "View job" link points, hence the `jobsBase` prop.

import { useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { uploadComfyInput, uploadComfySequence } from '@/lib/upload-stage'
import { parseComfyFormState } from '@/lib/comfy-form-state'

// ── Types (shape of /api/comfy/presets) ──────────────────────────────────────

interface ParamUi {
  label?: string; widget?: string; section?: string; order?: number
  min?: number; max?: number; step?: number; tooltip?: string; choices?: unknown[]
}
interface ParamSpec { node: string; path: string; default?: unknown; ui?: ParamUi }
interface ResolutionTier { label: string; short?: number; long?: number; width?: number; height?: number }
interface FileInputUi { param?: string; label?: string; tooltip?: string; filetypes?: [string, string][] }
interface PresetUi {
  title?: string
  primary_input?: FileInputUi
  secondary_input?: FileInputUi   // two-input presets (face image / mask)
  batch_input?: { label?: string; tooltip?: string }
  resolutions?: ResolutionTier[]
}
interface DocLink { label?: string; url: string }
interface About { what?: string; inputs?: string; key_knobs?: string }
interface Preset {
  key: string; description?: string; gpu?: string; mode?: string
  docs?: DocLink[] | string[] | string
  tags?: string[]
  about?: About
  ui?: PresetUi; params: Record<string, ParamSpec>
  output_prefix?: unknown          // present ⇒ outputs are named per-input (batchable)
  input_anchor?: { node: string }
  prompt_guide?: { url: string; tips?: string }   // model-specific prompting help (LTX-2 etc.)
}

// ── Batch mode (many inputs in one job) ──────────────────────────────────────

// Image-sequence frame extensions a folder can hold (DPX excluded — no loader yet).
const SEQ_EXTS = new Set(['exr', 'png', 'jpg', 'jpeg', 'tif', 'tiff'])
const extOf = (name: string) => { const i = name.lastIndexOf('.'); return i < 0 ? '' : name.slice(i + 1).toLowerCase() }

interface BatchItem {
  id:    string
  kind:  'video' | 'sequence'
  name:  string                    // video filename, or sequence-folder tag
  files: File[]
  bytes: number
  frameCount?: number              // sequence
  ext?:  string                    // sequence frame extension (exr/png/…)
}

/** A preset is batchable when it has a single primary clip input (video/image),
 *  names its outputs per-input (output_prefix), and takes no second file. */
function canBatch(p: Preset | null): boolean {
  if (!p) return false
  const hasPrimary = !!(p.params?.video || p.params?.image)
  return hasPrimary && !!p.output_prefix && !secondaryOf(p)
}

/** Sanitize a folder name into a filesystem-safe sequence tag. */
const seqTag = (name: string) => (name.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 60) || 'seq')

/** Normalize a preset's `docs` field to [{label,url}] (accepts a bare URL string or list). */
function docLinks(p: Preset | null): DocLink[] {
  const d = p?.docs
  if (!d) return []
  if (typeof d === 'string') return [{ url: d }]
  return d.map(x => (typeof x === 'string' ? { url: x } : x)).filter(x => x?.url)
}

/** The secondary uploaded input a preset wants (face image / mask), or null. */
function secondaryOf(p: Preset | null): Required<Pick<FileInputUi, 'param'>> & FileInputUi | null {
  if (!p) return null
  const sec = p.ui?.secondary_input
  if (sec) return { param: sec.param ?? 'face', label: sec.label, tooltip: sec.tooltip, filetypes: sec.filetypes }
  if (p.params?.mask) {
    return {
      param: 'mask', label: 'Mask image',
      tooltip: 'White = the region to inpaint/remove (static image).',
      filetypes: [['Image', '*.png *.jpg *.jpeg *.webp']],
    }
  }
  return null
}

/** Build an <input accept=…> string from a preset filetypes list. */
function acceptOf(filetypes: [string, string][] | undefined, fallback: string): string {
  if (!filetypes) return fallback
  const exts = filetypes.flatMap(([, pat]) => pat.split(/\s+/)).filter(p => p.startsWith('*.') && p !== '*.*')
  return exts.map(p => p.slice(1)).join(',') || fallback
}

const GPU_CHOICES: { key: string; label: string }[] = [
  { key: 'rtxpro6000',   label: 'RTX PRO 6000 96GB · fastest' },
  { key: 'l40s',         label: 'L40S 48GB' },
  { key: 'l4',           label: 'L4 24GB' },
  { key: 'a10',          label: 'A10 24GB · cheap' },
  { key: 't4',           label: 'T4 16GB · slow' },
  { key: 'rtxpro6000x8', label: '8× RTX PRO 6000 · ludicrous' },
]

interface PhaseEvent {
  phase: string; status?: string; jobId?: string; error?: string
  files?: number; kb?: number; ms?: number
}

export function ComfyForm({ jobsBase = '/jobs', cloneFromId = null }: { jobsBase?: string; cloneFromId?: string | null }) {
  const [presets, setPresets] = useState<Preset[] | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)
  const [clonedFrom, setClonedFrom] = useState<string | null>(null)
  const clonedRef = useRef(false)
  const [key, setKey]         = useState('')
  const [values, setValues]   = useState<Record<string, unknown>>({})
  const [gpu, setGpu]         = useState('rtxpro6000')
  const [mode, setMode]       = useState<'instant' | 'smart'>('instant')
  const [name, setName]       = useState('')

  const [file, setFile]       = useState<File | null>(null)
  const [file2, setFile2]     = useState<File | null>(null)   // secondary input (face/mask)
  const [upPct, setUpPct]     = useState<number | null>(null)
  const [probe, setProbe]     = useState<{ fps: number; frames: number; w: number; h: number } | null>(null)

  const [batchMode, setBatchMode]   = useState(false)
  const [batchItems, setBatchItems] = useState<BatchItem[]>([])

  const [events, setEvents]   = useState<PhaseEvent[]>([])
  const [busy, setBusy]       = useState(false)
  const [doneJob, setDoneJob] = useState<string | null>(null)
  const [err, setErr]         = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const file2Ref = useRef<HTMLInputElement | null>(null)
  const batchVideoRef  = useRef<HTMLInputElement | null>(null)
  const batchFolderRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    fetch('/api/comfy/presets')
      .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then((d: { presets: Preset[] }) => {
        setPresets(d.presets)
        if (d.presets[0]) selectPreset(d.presets[0])
      })
      .catch(e => setLoadErr(e instanceof Error ? e.message : 'failed to load presets'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ?clone=<jobId>: rehydrate the form from the source job's TUNET_FORM_STATE
  // stash (preset + prompt + knobs + gpu/mode/name). The input clip is NOT
  // cloned — the user re-picks it (same as the training clone not re-using the
  // dataset). Older jobs without a stash fall back to selecting their preset.
  useEffect(() => {
    if (!cloneFromId || !presets || clonedRef.current) return
    clonedRef.current = true
    ;(async () => {
      try {
        const res = await fetch(`/api/spark/jobs/${cloneFromId}`, { cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const job = data.job as { id: string; env?: Record<string, string> } | undefined
        if (!job) throw new Error('source job not found')

        const fs = parseComfyFormState(job.env?.TUNET_FORM_STATE)
        const presetKey = fs?.presetKey ?? job.env?.TUNET_PRESET
        const p = presets.find(x => x.key === presetKey)
        if (!p) { setErr(`Clone source used preset "${presetKey ?? '?'}", which isn't available here.`); return }

        // Start from the preset's defaults, then overlay the cloned values so
        // params the source didn't set keep sane defaults.
        const defaults: Record<string, unknown> = {}
        for (const [n, spec] of Object.entries(p.params)) {
          if (n === 'video' || n === 'mask' || n === 'face') continue
          if (spec.default !== undefined) defaults[n] = spec.default
        }
        setKey(p.key)
        setValues({ ...defaults, ...(fs?.values ?? {}) })
        setGpu(fs?.gpu || p.gpu || 'rtxpro6000')
        setMode((fs?.mode as 'instant' | 'smart') || (p.mode as 'instant' | 'smart') || 'instant')
        setName(fs?.name || '')
        setFile2(null); setBatchItems([]); setBatchMode(false)
        setDoneJob(null); setErr(null); setEvents([])
        setClonedFrom(job.id)
      } catch (e) {
        setErr(e instanceof Error ? `Clone failed: ${e.message}` : 'Clone failed')
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot once presets load
  }, [presets, cloneFromId])

  const preset = useMemo(() => presets?.find(p => p.key === key) ?? null, [presets, key])

  function selectPreset(p: Preset) {
    setKey(p.key)
    const v: Record<string, unknown> = {}
    for (const [n, spec] of Object.entries(p.params)) {
      if (n === 'video' || n === 'mask' || n === 'face') continue
      if (spec.default !== undefined) v[n] = spec.default
    }
    setValues(v)
    setFile2(null)
    setBatchItems([]); setBatchMode(false)
    setGpu(p.gpu || 'rtxpro6000')
    setMode((p.mode as 'instant' | 'smart') || 'instant')
    setDoneJob(null); setErr(null); setEvents([])
    setClonedFrom(null)   // a manual preset switch is no longer "this clone"
  }

  // Match the desktop tool: probe the picked clip and fill fps + frame count so
  // the output matches the source (web can't ffprobe, so we read duration from a
  // <video> + sample frame timestamps for fps; degrades to no-op if undetectable).
  async function onPickClip(f: File) {
    setProbe(null)
    const info = await probeClip(f)
    if (!info) return
    setProbe(info)
    setValues(v => {
      const next = { ...v }
      if (preset?.params.fps && info.fps > 0) next.fps = info.fps
      if (preset?.params.length && info.frames > 0) {
        next.length = Math.max(9, Math.round((info.frames - 1) / 8) * 8 + 1) // LTX needs 8n+1
      }
      return next
    })
  }

  // group params by section, hide structural ones (video/mask handled by upload)
  const sections = useMemo(() => {
    const out: Record<string, [string, ParamSpec][]> = {}
    if (!preset) return out
    for (const [n, spec] of Object.entries(preset.params)) {
      if (n === 'video' || n === 'mask' || n === 'face' || !spec.ui) continue
      const sec = spec.ui.section || 'Main'
      ;(out[sec] ??= []).push([n, spec])
    }
    for (const sec of Object.keys(out)) out[sec].sort((a, b) => (a[1].ui?.order ?? 99) - (b[1].ui?.order ?? 99))
    return out
  }, [preset])

  const sizeKeys = useMemo<[string, string, 'short' | 'long' | 'width' | 'height', 'short' | 'long' | 'width' | 'height'] | null>(() => {
    if (!preset) return null
    if (preset.params.short_edge && preset.params.long_edge) return ['short_edge', 'long_edge', 'short', 'long']
    if (preset.params.width && preset.params.height)         return ['width', 'height', 'width', 'height']
    return null
  }, [preset])

  const acceptStr = useMemo(() => acceptOf(preset?.ui?.primary_input?.filetypes, 'video/*'), [preset])
  const secondary = useMemo(() => secondaryOf(preset), [preset])
  const accept2   = useMemo(() => acceptOf(secondary?.filetypes, 'image/*'), [secondary])
  const allowBatch = useMemo(() => canBatch(preset), [preset])

  // ── batch input collection (videos + image-sequence folders) ───────────────
  function addVideos(list: FileList | null) {
    if (!list || list.length === 0) return
    const incoming = Array.from(list)
    setBatchItems(prev => {
      const names = new Set(prev.map(i => i.name))
      const add: BatchItem[] = []
      for (const f of incoming) {
        if (names.has(f.name)) continue
        names.add(f.name)
        add.push({ id: `v_${f.name}_${f.size}`, kind: 'video', name: f.name, files: [f], bytes: f.size })
      }
      return [...prev, ...add]
    })
  }

  function addFolder(list: FileList | null) {
    if (!list || list.length === 0) return
    // Group by the top folder segment of webkitRelativePath; keep only frame files.
    const groups = new Map<string, File[]>()
    for (const f of Array.from(list)) {
      if (!SEQ_EXTS.has(extOf(f.name))) continue
      const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
      const seg = rel.split('/')[0] || 'sequence'
      const g = groups.get(seg) ?? []
      g.push(f); groups.set(seg, g)
    }
    if (groups.size === 0) { setErr('No EXR / PNG / JPG / TIFF frames found in that folder.'); return }
    setErr(null)
    setBatchItems(prev => {
      const names = new Set(prev.map(i => i.name))
      const add: BatchItem[] = []
      for (const [seg, files] of groups) {
        const tag = seqTag(seg)
        if (names.has(tag) || files.length === 0) continue
        names.add(tag)
        files.sort((a, b) => a.name.localeCompare(b.name))
        add.push({ id: `s_${tag}`, kind: 'sequence', name: tag, files,
                   bytes: files.reduce((s, f) => s + f.size, 0), frameCount: files.length, ext: extOf(files[0].name) })
      }
      return [...prev, ...add]
    })
  }

  const removeItem = (id: string) => setBatchItems(prev => prev.filter(i => i.id !== id))

  async function onSubmit() {
    if (!preset) return
    if (batchMode) {
      if (batchItems.length === 0) { setErr('Add at least one clip or image-sequence folder.'); return }
    } else if (secondary && !file2) {
      setErr(`Pick a ${secondary.label?.toLowerCase() || 'reference file'} first.`); return
    }
    setBusy(true); setErr(null); setDoneJob(null); setEvents([])
    try {
      let body: Record<string, unknown>
      if (batchMode) {
        // Upload every input into ONE stage: clips under comfy_input, each
        // sequence folder under comfy_seq/<tag>. Progress spans all of them.
        const totalBytes = batchItems.reduce((s, i) => s + i.bytes, 0) || 1
        let stageId: string | undefined
        let base = 0
        setUpPct(0)
        for (const it of batchItems) {
          const onProg = (sent: number) => setUpPct(Math.round(((base + sent) / totalBytes) * 100))
          if (it.kind === 'video') {
            const r = await uploadComfyInput(it.files[0], onProg, { role: 'comfy_input', stageId })
            stageId = r.stageId
          } else {
            const r = await uploadComfySequence(it.files, it.name, stageId, onProg)
            stageId = r.stageId
          }
          base += it.bytes
        }
        setUpPct(null)
        body = { presetKey: preset.key, values: { ...values, __name: name || undefined }, stageId, batch: true, gpu, mode }
      } else {
        let stageId: string | undefined
        let inputName: string | undefined
        let secondaryName: string | undefined
        if (file) {
          setUpPct(0)
          const r = await uploadComfyInput(file, (sent, total) => setUpPct(Math.round((sent / total) * 100)))
          stageId = r.stageId; inputName = r.filename
          setUpPct(null)
        }
        if (secondary && file2) {
          // drop the secondary file into the SAME stage (a second role dir) so the
          // submit route resolves both from one stageId
          setUpPct(0)
          const r2 = await uploadComfyInput(file2, (sent, total) => setUpPct(Math.round((sent / total) * 100)),
            { role: 'comfy_input2', stageId })
          stageId = r2.stageId; secondaryName = r2.filename
          setUpPct(null)
        }
        body = { presetKey: preset.key, values: { ...values, __name: name || undefined }, stageId, inputName, secondaryName, gpu, mode }
      }
      const res = await fetch('/api/comfy/submit', {
        method:  'POST',
        headers: { 'content-type': 'application/json' },
        body:    JSON.stringify(body),
      })
      if (!res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const blocks = buf.split('\n\n'); buf = blocks.pop() ?? ''
        for (const b of blocks) {
          const line = b.split('\n').find(l => l.startsWith('data: '))
          if (!line) continue
          const evt = JSON.parse(line.slice(6)) as PhaseEvent
          setEvents(prev => [...prev, evt])
          if (evt.phase === 'done' && evt.jobId) setDoneJob(evt.jobId)
          if (evt.phase === 'error') setErr(evt.error ?? 'submit failed')
        }
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'submit failed')
    } finally {
      setBusy(false); setUpPct(null)
    }
  }

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div>
      <div className="flex items-center gap-2.5 mb-2">
        <h1 className="text-xl font-bold text-[#111827]">EZ-Comfy</h1>
        <span className="text-xs font-semibold text-[#7E3AF2] bg-[#faf5ff] border border-[#e9d5ff] rounded px-2 py-0.5">Beta</span>
      </div>
      <p className="text-sm text-[#6b7280] mb-6 max-w-2xl">
        Run ComfyUI workflows on Spark. Pick a preset, drop in your clip, tune the knobs, submit.
      </p>

      {loadErr && <Card><p className="text-sm text-[#DC2626]">Couldn&apos;t load presets: {loadErr}</p></Card>}
      {!presets && !loadErr && <Card><p className="text-sm text-[#6b7280]">Loading presets…</p></Card>}

      {preset && (
        <div className="space-y-4">
          {/* Cloned-from notice — settings are pre-filled; the clip is re-picked. */}
          {clonedFrom && (
            <div className="flex items-center justify-between gap-3 rounded-lg border border-[#e9d5ff] bg-[#faf5ff] px-4 py-2.5 text-sm text-[#6b21a8]">
              <span>
                Cloned from{' '}
                <Link href={`${jobsBase}/${clonedFrom}`} className="font-mono text-xs text-[#7E3AF2] hover:underline">
                  {clonedFrom.slice(0, 8)}
                </Link>
                {' '}— settings pre-filled. Re-select your input clip{secondary ? ' and reference image' : ''} before submitting.
              </span>
              <button type="button" onClick={() => setClonedFrom(null)} aria-label="Dismiss"
                className="flex-shrink-0 text-[#9ca3af] hover:text-[#6b21a8]">✕</button>
            </div>
          )}

          {/* Preset picker */}
          <Card>
            <Label>Workflow preset</Label>
            <select value={key} onChange={e => { const p = presets!.find(x => x.key === e.target.value); if (p) selectPreset(p) }}
              className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white focus:border-[#7E3AF2] focus:outline-none">
              {presets!.map(p => <option key={p.key} value={p.key}>{p.ui?.title || p.key}</option>)}
            </select>
          </Card>

          {/* About this workflow — plain-language summary + canonical docs link */}
          {(preset.about || preset.tags?.length || docLinks(preset).length > 0) && (
            <Card>
              <h2 className="text-base font-semibold text-[#111827] mb-3.5">About this workflow</h2>
              <dl className="space-y-4">
                {preset.about?.what && (
                  <div>
                    <dt className="text-[11px] font-semibold uppercase tracking-wide text-[#6b7280] mb-1">What it does</dt>
                    <dd className="text-[15px] leading-relaxed text-[#374151]">{preset.about.what}</dd>
                  </div>
                )}
                {preset.about?.inputs && (
                  <div>
                    <dt className="text-[11px] font-semibold uppercase tracking-wide text-[#6b7280] mb-1">Inputs</dt>
                    <dd className="text-[15px] leading-relaxed text-[#374151]">{preset.about.inputs}</dd>
                  </div>
                )}
                {preset.about?.key_knobs && (
                  <div>
                    <dt className="text-[11px] font-semibold uppercase tracking-wide text-[#6b7280] mb-1">Key settings</dt>
                    <dd className="text-[15px] leading-relaxed text-[#374151]">{preset.about.key_knobs}</dd>
                  </div>
                )}
              </dl>
              {preset.tags && preset.tags.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {preset.tags.map(t => (
                    <span key={t} className="text-xs font-medium text-[#4b5563] bg-[#f3f4f6] border border-[#e5e7eb] rounded-full px-2.5 py-1">
                      {t}
                    </span>
                  ))}
                </div>
              )}
              {docLinks(preset).length > 0 && (
                <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1">
                  {docLinks(preset).map(d => (
                    <a key={d.url} href={d.url} target="_blank" rel="noopener noreferrer"
                      className="text-sm font-semibold text-[#7E3AF2] hover:underline">
                      {d.label || 'Documentation'} ↗
                    </a>
                  ))}
                </div>
              )}
            </Card>
          )}

          {/* Single ⟷ Batch toggle (only for presets that name outputs per-input) */}
          {allowBatch && (
            <div className="flex items-center gap-1 rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-1 w-fit">
              {([['single', 'One clip'], ['batch', 'Batch (many)']] as const).map(([m, lbl]) => (
                <button key={m} type="button" onClick={() => setBatchMode(m === 'batch')}
                  className={`px-3.5 py-1.5 text-xs font-semibold rounded-md transition-colors ${
                    (m === 'batch') === batchMode ? 'bg-white text-[#7E3AF2] shadow-sm' : 'text-[#6b7280] hover:text-[#374151]'}`}>
                  {lbl}
                </button>
              ))}
            </div>
          )}

          {/* Input clip (single mode) */}
          {!batchMode && (
          <Card>
            <Label tip={preset.ui?.primary_input?.tooltip}>{preset.ui?.primary_input?.label || 'Input clip'}</Label>
            <div className="flex items-center gap-3">
              <input ref={fileRef} type="file" accept={acceptStr} className="hidden"
                onChange={e => { const f = e.target.files?.[0] ?? null; setFile(f); setProbe(null); if (f) void onPickClip(f) }} />
              <button type="button" onClick={() => fileRef.current?.click()}
                className="px-3 py-1.5 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff]">
                {file ? 'Replace clip' : 'Browse…'}
              </button>
              <span className="text-xs text-[#6b7280] truncate">
                {file ? `${file.name} (${(file.size / 1048576).toFixed(1)} MB)` : 'No clip selected'}
              </span>
            </div>
            {probe && (
              <p className="mt-2 text-xs text-[#16A34A]">
                Detected {probe.w}×{probe.h}
                {probe.fps > 0 ? ` · ${probe.fps} fps` : ''}
                {probe.frames > 0 ? ` · ~${probe.frames} frames` : ''}
                {' '}— fps/frames filled in below to match the clip.
              </p>
            )}
            {upPct !== null && (
              <div className="mt-2 h-1.5 bg-[#e5e7eb] rounded overflow-hidden">
                <div className="h-full bg-[#7E3AF2] transition-all" style={{ width: `${upPct}%` }} />
              </div>
            )}
          </Card>
          )}

          {/* Batch inputs (batch mode) */}
          {batchMode && (
            <BatchPanel
              items={batchItems}
              label={preset.ui?.batch_input?.label || 'Batch inputs (videos + image-sequence folders)'}
              tip={preset.ui?.batch_input?.tooltip ||
                'Add any number of video clips and/or folders of frames (EXR / PNG / JPG / TIFF). They all process in one job on a single warm GPU — far cheaper than one job each — and every output is named after its input.'}
              videoAccept={acceptStr}
              upPct={upPct}
              videoRef={batchVideoRef}
              folderRef={batchFolderRef}
              onAddVideos={addVideos}
              onAddFolder={addFolder}
              onRemove={removeItem}
              onClear={() => setBatchItems([])}
            />
          )}

          {/* Secondary input (face image / mask) — two-input presets */}
          {secondary && (
            <Card>
              <Label tip={secondary.tooltip}>{secondary.label || 'Reference image'}</Label>
              <div className="flex items-center gap-3">
                <input ref={file2Ref} type="file" accept={accept2} className="hidden"
                  onChange={e => setFile2(e.target.files?.[0] ?? null)} />
                <button type="button" onClick={() => file2Ref.current?.click()}
                  className="px-3 py-1.5 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff]">
                  {file2 ? 'Replace' : 'Browse…'}
                </button>
                <span className="text-xs text-[#6b7280] truncate">
                  {file2 ? `${file2.name} (${(file2.size / 1048576).toFixed(1)} MB)` : 'Required'}
                </span>
              </div>
            </Card>
          )}

          {/* Param sections */}
          {Object.entries(sections).map(([sec, rows]) => (
            <Card key={sec}>
              <h2 className="text-sm font-semibold text-[#111827] mb-3">{sec}</h2>
              {sec === 'Advanced' && sizeKeys && preset.ui?.resolutions && (
                <ResolutionRow tiers={preset.ui.resolutions} sizeKeys={sizeKeys}
                  onPick={(a, b) => setValues(v => ({ ...v, [sizeKeys[0]]: a, [sizeKeys[1]]: b }))} />
              )}
              <div className="space-y-3">
                {rows.map(([n, spec]) => (
                  <div key={n}>
                    <ParamRow name={n} spec={spec} value={values[n]}
                      onChange={val => setValues(v => ({ ...v, [n]: val }))} />
                    {n === 'prompt' && preset.prompt_guide && <PromptGuideNote guide={preset.prompt_guide} />}
                  </div>
                ))}
              </div>
            </Card>
          ))}

          {/* Compute */}
          <Card>
            <h2 className="text-sm font-semibold text-[#111827] mb-3">Compute</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <Label>GPU</Label>
                <select value={gpu} onChange={e => setGpu(e.target.value)}
                  className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white">
                  {GPU_CHOICES.map(g => <option key={g.key} value={g.key}>{g.label}</option>)}
                </select>
              </div>
              <div>
                <Label>Mode</Label>
                <select value={mode} onChange={e => setMode(e.target.value as 'instant' | 'smart')}
                  className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white">
                  <option value="instant">Instant</option>
                  <option value="smart">Smart (cheaper, preemptible)</option>
                </select>
              </div>
              <div>
                <Label>Job name (optional)</Label>
                <input value={name} onChange={e => setName(e.target.value)} placeholder={`comfy-${preset.key}`}
                  className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white" />
              </div>
            </div>
          </Card>

          {/* Submit */}
          <div className="flex items-center gap-3">
            <button type="button" disabled={busy || (batchMode && batchItems.length === 0)} onClick={() => void onSubmit()}
              className="rounded-xl bg-[#ae69f4] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#7E3AF2] disabled:opacity-60">
              {busy
                ? (upPct !== null ? `Uploading ${upPct}%…` : 'Submitting…')
                : batchMode
                  ? (batchItems.length
                      ? `Submit ${batchItems.length} input${batchItems.length === 1 ? '' : 's'} to Spark`
                      : 'Add inputs to submit')
                  : 'Submit to Spark'}
            </button>
            {doneJob && (
              <Link href={`${jobsBase}/${doneJob}`} className="text-sm font-semibold text-[#7E3AF2] hover:underline">
                View job &amp; download outputs →
              </Link>
            )}
          </div>

          {/* Progress / errors */}
          {(events.length > 0 || err) && (
            <Card>
              <div className="space-y-1 text-xs font-mono">
                {events.map((e, i) => (
                  <div key={i} className={e.phase === 'error' ? 'text-[#DC2626]' : 'text-[#374151]'}>
                    {e.phase === 'error'
                      ? `✗ ${e.error}`
                      : `${e.status === 'done' ? '✓' : '•'} ${e.phase}${e.status ? ` — ${e.status}` : ''}` +
                        (e.jobId ? `  ${e.jobId}` : '') + (e.kb ? `  ${e.kb} KB` : '')}
                  </div>
                ))}
                {err && !events.some(e => e.phase === 'error') && <div className="text-[#DC2626]">✗ {err}</div>}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}

// ── clip probe (client-side; matches the desktop ffprobe behavior) ────────────

async function probeClip(file: File): Promise<{ fps: number; frames: number; w: number; h: number } | null> {
  const url = URL.createObjectURL(file)
  const v = document.createElement('video')
  v.muted = true
  v.preload = 'auto'
  v.src = url
  try {
    await new Promise<void>((res, rej) => {
      v.onloadedmetadata = () => res()
      v.onerror = () => rej(new Error('metadata load failed'))
      setTimeout(() => rej(new Error('metadata timeout')), 8000)
    })
    const dur = v.duration
    const fps = await detectFps(v)
    const frames = fps > 0 && Number.isFinite(dur) && dur > 0 ? Math.round(dur * fps) : 0
    return { fps, frames, w: v.videoWidth, h: v.videoHeight }
  } catch {
    return null
  } finally {
    URL.revokeObjectURL(url)
  }
}

// fps isn't exposed by HTMLVideoElement, so sample frame timestamps via
// requestVideoFrameCallback (play muted briefly) and take the median delta.
function detectFps(v: HTMLVideoElement): Promise<number> {
  type VFC = HTMLVideoElement & {
    requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number
  }
  const vv = v as VFC
  if (typeof vv.requestVideoFrameCallback !== 'function') return Promise.resolve(0)
  return new Promise(resolve => {
    const times: number[] = []
    let done = false
    const finish = () => {
      if (done) return
      done = true
      try { v.pause() } catch { /* ignore */ }
      const deltas = times.slice(1).map((t, i) => t - times[i]).filter(d => d > 0).sort((a, b) => a - b)
      const med = deltas.length ? deltas[Math.floor(deltas.length / 2)] : 0
      resolve(med > 0 ? Math.round(1 / med) : 0)
    }
    const cb = (_now: number, meta: { mediaTime: number }) => {
      times.push(meta.mediaTime)
      if (times.length >= 12) { finish(); return }
      vv.requestVideoFrameCallback!(cb)
    }
    vv.requestVideoFrameCallback!(cb)
    v.play().catch(() => finish())   // muted autoplay; if blocked, give up gracefully
    setTimeout(finish, 6000)         // safety cap
  })
}

// ── small components ──────────────────────────────────────────────────────────

function Card({ children }: { children: React.ReactNode }) {
  return <div className="bg-white border border-[#e5e7eb] rounded-xl p-5 card-shadow">{children}</div>
}

function fmtBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1048576).toFixed(1)} MB`
  return `${(n / 1073741824).toFixed(2)} GB`
}

/** Model-specific prompting help shown under the prompt field: a clickable link to
 *  the official guide, with the key tips on hover. Driven by preset.prompt_guide. */
function PromptGuideNote({ guide }: { guide: { url: string; tips?: string } }) {
  return (
    <p className="mt-1.5 flex items-center gap-1 text-xs text-[#6b7280]">
      {guide.tips && (
        <span className="relative inline-flex group" tabIndex={0}>
          <span className="text-[#9ca3af] cursor-help">ⓘ</span>
          <span role="tooltip"
            className="pointer-events-none invisible opacity-0 group-hover:visible group-hover:opacity-100
                       group-focus-within:visible group-focus-within:opacity-100 transition-opacity duration-100 delay-150
                       absolute left-0 top-5 z-50 bg-[#1F2937] text-[#F3F4F6] text-xs font-normal leading-relaxed
                       rounded-md shadow-lg px-3 py-2 w-72 max-w-[18rem] whitespace-pre-wrap">
            {guide.tips}
          </span>
        </span>
      )}
      <span>Tips:</span>
      <a href={guide.url} target="_blank" rel="noopener noreferrer"
        className="font-semibold text-[#7E3AF2] hover:underline">
        prompting guide ↗
      </a>
    </p>
  )
}

/** Batch input collector: add many clips and/or image-sequence folders, see them
 *  listed, remove any. Drop video files onto the card, or use the buttons. */
function BatchPanel({
  items, label, tip, videoAccept, upPct, videoRef, folderRef,
  onAddVideos, onAddFolder, onRemove, onClear,
}: {
  items: BatchItem[]; label: string; tip: string; videoAccept: string; upPct: number | null
  videoRef: React.MutableRefObject<HTMLInputElement | null>
  folderRef: React.MutableRefObject<HTMLInputElement | null>
  onAddVideos: (l: FileList | null) => void
  onAddFolder: (l: FileList | null) => void
  onRemove: (id: string) => void
  onClear: () => void
}) {
  const [drag, setDrag] = useState(false)
  const clips = items.filter(i => i.kind === 'video')
  const seqs  = items.filter(i => i.kind === 'sequence')
  const totalBytes = items.reduce((s, i) => s + i.bytes, 0)
  const empty = items.length === 0

  return (
    <Card>
      <div className="flex items-center justify-between gap-3 mb-2">
        <Label tip={tip}>
          {label}{!empty && <span className="ml-1.5 font-normal text-[#9ca3af] tabular-nums">· {items.length}</span>}
        </Label>
        {!empty && (
          <button type="button" onClick={onClear}
            className="text-xs text-[#9ca3af] hover:text-[#DC2626] font-semibold cursor-pointer">
            Clear all
          </button>
        )}
      </div>

      <input ref={videoRef} type="file" accept={videoAccept} multiple className="hidden"
        onChange={e => { onAddVideos(e.target.files); e.target.value = '' }} />
      <input
        ref={el => { folderRef.current = el; if (el) el.setAttribute('webkitdirectory', '') }}
        type="file" multiple className="hidden"
        onChange={e => { onAddFolder(e.target.files); e.target.value = '' }} />

      {/* Drop zone + add buttons — roomy when empty, compact once inputs are added */}
      <div
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); onAddVideos(e.dataTransfer.files) }}
        className={`rounded-lg border-2 border-dashed text-center transition-colors ${empty ? 'px-4 py-6' : 'px-3 py-3'} ${
          drag ? 'border-[#7E3AF2] bg-[#faf5ff]' : 'border-[#e5e7eb] bg-[#fafafa]'}`}
      >
        {empty && (
          <p className="text-sm text-[#6b7280] mb-1">Drop video files here, or add inputs</p>
        )}
        {empty && (
          <p className="text-xs text-[#9ca3af] mb-3">
            Videos (.mp4 .mov …) and/or folders of frames (EXR · PNG · JPG · TIFF)
          </p>
        )}
        <div className="flex items-center justify-center gap-2">
          <button type="button" onClick={() => videoRef.current?.click()}
            className="px-3 py-1.5 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff] cursor-pointer">
            + Video files…
          </button>
          <button type="button" onClick={() => folderRef.current?.click()}
            className="px-3 py-1.5 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff] cursor-pointer">
            + Image-sequence folder…
          </button>
          {!empty && <span className="text-xs text-[#9ca3af] hidden sm:inline">or drop video files</span>}
        </div>
      </div>

      {/* Item list */}
      {!empty && (
        <ul className="mt-3 divide-y divide-[#f3f4f6] border border-[#e5e7eb] rounded-lg overflow-hidden max-h-72 overflow-y-auto">
          {items.map(it => (
            <li key={it.id} className="flex items-center gap-3 px-3 py-2 hover:bg-[#fafafa]">
              <span className={`flex-shrink-0 w-12 text-center text-[10px] font-bold uppercase tracking-wide rounded px-1 py-0.5 ${
                it.kind === 'video' ? 'bg-[#eef2ff] text-[#4f46e5]' : 'bg-[#ecfdf5] text-[#059669]'}`}>
                {it.kind === 'video' ? 'clip' : (it.ext || 'seq')}
              </span>
              <span className="flex-1 min-w-0 text-sm text-[#374151] truncate" title={it.name}>{it.name}</span>
              <span className="flex-shrink-0 text-xs text-[#9ca3af] tabular-nums">
                {it.kind === 'sequence' ? `${it.frameCount} frames · ` : ''}{fmtBytes(it.bytes)}
              </span>
              <button type="button" onClick={() => onRemove(it.id)} aria-label={`Remove ${it.name}`}
                className="flex-shrink-0 text-[#9ca3af] hover:text-[#DC2626] leading-none p-1 -m-1 cursor-pointer">✕</button>
            </li>
          ))}
        </ul>
      )}

      {/* Summary + progress */}
      {!empty && (
        <p className="mt-2 text-xs text-[#16A34A]">
          {clips.length > 0 && `${clips.length} clip${clips.length === 1 ? '' : 's'}`}
          {clips.length > 0 && seqs.length > 0 && ' · '}
          {seqs.length > 0 && `${seqs.length} sequence${seqs.length === 1 ? '' : 's'}`}
          {' · '}{fmtBytes(totalBytes)} — one job, model loads once; each output is named after its input.
        </p>
      )}
      {upPct !== null && (
        <div className="mt-2 h-1.5 bg-[#e5e7eb] rounded overflow-hidden" role="progressbar"
          aria-valuenow={upPct} aria-valuemin={0} aria-valuemax={100} aria-label="Upload progress">
          <div className="h-full bg-[#7E3AF2] transition-all" style={{ width: `${upPct}%` }} />
        </div>
      )}
    </Card>
  )
}

function Label({ children, tip }: { children: React.ReactNode; tip?: string }) {
  return (
    <label className="flex items-center gap-1 text-xs font-semibold text-[#374151] mb-1.5">
      <span>{children}</span>
      {tip && (
        <span className="relative inline-flex group" tabIndex={0}>
          <span className="text-[#9ca3af] cursor-help">ⓘ</span>
          <span
            role="tooltip"
            className="pointer-events-none invisible opacity-0 group-hover:visible group-hover:opacity-100
                       group-focus-within:visible group-focus-within:opacity-100 transition-opacity duration-100 delay-150
                       absolute left-0 top-5 z-50 bg-[#1F2937] text-[#F3F4F6] text-xs font-normal leading-relaxed
                       rounded-md shadow-lg px-3 py-2 w-72 max-w-[18rem] whitespace-pre-wrap"
          >
            {tip}
          </span>
        </span>
      )}
    </label>
  )
}

function ParamRow({ name, spec, value, onChange }: {
  name: string; spec: ParamSpec; value: unknown; onChange: (v: unknown) => void
}) {
  const ui = spec.ui ?? {}
  const label = ui.label || name
  if (ui.widget === 'multiline') {
    return (
      <div>
        <Label tip={ui.tooltip}>{label}</Label>
        <textarea value={String(value ?? '')} onChange={e => onChange(e.target.value)} rows={3}
          className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white focus:border-[#7E3AF2] focus:outline-none" />
      </div>
    )
  }
  if (ui.widget === 'combo') {
    return (
      <div>
        <Label tip={ui.tooltip}>{label}</Label>
        <select value={String(value ?? '')} onChange={e => onChange(e.target.value)}
          className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white">
          {(ui.choices ?? []).map((c, i) => <option key={i} value={String(c)}>{String(c)}</option>)}
        </select>
      </div>
    )
  }
  if (ui.widget === 'slider' || ui.widget === 'number') {
    const num = typeof value === 'number' ? value : Number(value ?? ui.min ?? 0)
    return (
      <div>
        <Label tip={ui.tooltip}>{label}{ui.widget === 'slider' ? `: ${num}` : ''}</Label>
        <input type={ui.widget === 'slider' ? 'range' : 'number'} value={num}
          min={ui.min} max={ui.max} step={ui.step}
          onChange={e => onChange(Number(e.target.value))}
          className={ui.widget === 'slider' ? 'w-full' : 'w-40 text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white'} />
      </div>
    )
  }
  return (
    <div>
      <Label tip={ui.tooltip}>{label}</Label>
      <input value={String(value ?? '')} onChange={e => onChange(e.target.value)}
        className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white" />
    </div>
  )
}

function ResolutionRow({ tiers, sizeKeys, onPick }: {
  tiers: ResolutionTier[]
  sizeKeys: [string, string, 'short' | 'long' | 'width' | 'height', 'short' | 'long' | 'width' | 'height']
  onPick: (a: number, b: number) => void
}) {
  const [, , fa, fb] = sizeKeys
  return (
    <div className="mb-3">
      <Label tip="Known-good sizes. Output orientation follows the source clip for auto-orienting graphs.">Resolution</Label>
      <select defaultValue="" onChange={e => {
        const t = tiers.find(x => x.label === e.target.value)
        if (t) onPick(Number(t[fa]), Number(t[fb]))
      }} className="w-full text-sm border border-[#e5e7eb] rounded-lg px-3 py-2 bg-white">
        <option value="" disabled>Pick a size tier…</option>
        {tiers.map(t => <option key={t.label} value={t.label}>{t.label}</option>)}
      </select>
    </div>
  )
}
