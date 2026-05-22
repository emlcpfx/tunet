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
import { uploadComfyInput } from '@/lib/upload-stage'

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
}

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

export function ComfyForm({ jobsBase = '/jobs' }: { jobsBase?: string }) {
  const [presets, setPresets] = useState<Preset[] | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)
  const [key, setKey]         = useState('')
  const [values, setValues]   = useState<Record<string, unknown>>({})
  const [gpu, setGpu]         = useState('rtxpro6000')
  const [mode, setMode]       = useState<'instant' | 'smart'>('instant')
  const [name, setName]       = useState('')

  const [file, setFile]       = useState<File | null>(null)
  const [file2, setFile2]     = useState<File | null>(null)   // secondary input (face/mask)
  const [upPct, setUpPct]     = useState<number | null>(null)
  const [probe, setProbe]     = useState<{ fps: number; frames: number; w: number; h: number } | null>(null)

  const [events, setEvents]   = useState<PhaseEvent[]>([])
  const [busy, setBusy]       = useState(false)
  const [doneJob, setDoneJob] = useState<string | null>(null)
  const [err, setErr]         = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const file2Ref = useRef<HTMLInputElement | null>(null)

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
    setGpu(p.gpu || 'rtxpro6000')
    setMode((p.mode as 'instant' | 'smart') || 'instant')
    setDoneJob(null); setErr(null); setEvents([])
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

  async function onSubmit() {
    if (!preset) return
    if (secondary && !file2) { setErr(`Pick a ${secondary.label?.toLowerCase() || 'reference file'} first.`); return }
    setBusy(true); setErr(null); setDoneJob(null); setEvents([])
    try {
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
      const res = await fetch('/api/comfy/submit', {
        method:  'POST',
        headers: { 'content-type': 'application/json' },
        body:    JSON.stringify({
          presetKey: preset.key,
          values:    { ...values, __name: name || undefined },
          stageId, inputName, secondaryName, gpu, mode,
        }),
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

          {/* Input clip */}
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
                  <ParamRow key={n} name={n} spec={spec} value={values[n]}
                    onChange={val => setValues(v => ({ ...v, [n]: val }))} />
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
            <button type="button" disabled={busy} onClick={() => void onSubmit()}
              className="rounded-xl bg-[#ae69f4] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#7E3AF2] disabled:opacity-60">
              {busy ? (upPct !== null ? `Uploading ${upPct}%…` : 'Submitting…') : 'Submit to Spark'}
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
