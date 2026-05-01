'use client'

/**
 * /demo/jobs/new — preset-driven training job submitter.
 *
 * 4-step layout (mirrors Spark/frontend/new-job.html):
 *   1. Job Name
 *   2. Training Data (ShareSync paths or /input/data)
 *   3. Preset (cards) + Recommended settings + Advanced overrides
 *   4. Review & Submit (GPU + cost estimate + Submit CTA)
 *
 * On submit: POSTs to /api/spark/training-jobs which packs the tunet/ source
 * tarball server-side and ships it to Spark. On 201 → redirect to job detail.
 */

import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input, FormRow } from '@/components/ui/input'
import { FolderPicker, type FolderPickerResult } from '@/components/spark/folder-picker'
import { SubmitProgress, type SubmitEvent } from '@/components/spark/submit-progress'
import {
  PRESETS, type Preset, type PresetKey, type AdvancedOverrides,
  estimateRuntimeHours, pricePerHour,
} from '@/lib/spark-presets'

interface GpuOption {
  key:   string
  sku:   string
  label: string
  vram:  string
  badge?: 'recommended' | 'cheap' | 'fastest' | 'expensive'
}

const GPU_OPTIONS: GpuOption[] = [
  { key: 't4',           sku: 'g4dn.xlarge',  label: 'T4',           vram: '16GB',   badge: 'cheap'       },
  { key: 'a10',          sku: 'g5.xlarge',    label: 'A10',          vram: '24GB'                          },
  { key: 'l4',           sku: 'g6.2xlarge',   label: 'L4',           vram: '24GB'                          },
  { key: 'l40s',         sku: 'g6e.4xlarge',  label: 'L40S',         vram: '48GB',   badge: 'recommended' },
  { key: 'a10x4',        sku: 'g5.24xlarge',  label: '4× A10',       vram: '24GB'                          },
  { key: 'l40sx4',       sku: 'g6e.12xlarge', label: '4× L40S',      vram: '48GB',   badge: 'expensive'   },
  { key: 'rtxpro6000',   sku: 'g7e.xlarge',   label: 'RTX PRO 6000', vram: '96GB',   badge: 'fastest'     },
  { key: 'rtxpro6000x8', sku: 'g7e.48xlarge', label: '8× RTX PRO',   vram: '96GB',   badge: 'expensive'   },
]

const RESOLUTIONS = [256, 384, 512, 768, 1024]
const MODEL_SIZES = [32, 64, 128, 256]
const BATCH_SIZES = [1, 2, 4, 8, 16]
const LOSSES: { value: AdvancedOverrides['loss']; label: string }[] = [
  { value: 'l1',       label: 'L1 (default)' },
  { value: 'l1+lpips', label: 'L1 + LPIPS' },
  { value: 'weighted', label: 'Weighted (L1 + L2 + LPIPS)' },
  { value: 'bce+dice', label: 'BCE + Dice (matte/mask)' },
]

export default function NewJobPage() {
  const router = useRouter()

  // Form state
  const [name,    setName]    = useState('')
  const [srcDir,  setSrcDir]  = useState('/input/data/src')
  const [dstDir,  setDstDir]  = useState('/input/data/dst')
  const [valSrcDir, setValSrcDir] = useState<string>('')
  const [valDstDir, setValDstDir] = useState<string>('')
  const [preset,  setPreset]  = useState<PresetKey>('beauty')
  const [gpuKey,  setGpuKey]  = useState<string>('l40s')
  const [advanced, setAdvanced] = useState<AdvancedOverrides>({})
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [maxSteps, setMaxSteps] = useState<number>(15000)
  const [pairs, setPairs] = useState<number>(200)   // user-supplied or auto-detected from folder pick
  const [pairsAutoDetected, setPairsAutoDetected] = useState(false)
  const [idleHold, setIdleHold] = useState(0)
  const [showManualPaths, setShowManualPaths] = useState(false)
  const [picked, setPicked] = useState<FolderPickerResult | null>(null)

  // UX state
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Streaming submit progress
  const [progressOpen, setProgressOpen] = useState(false)
  const [progressEvents, setProgressEvents] = useState<SubmitEvent[]>([])
  const [progressError,  setProgressError]  = useState<string | null>(null)
  const [progressStartTs, setProgressStartTs] = useState<number>(0)
  const [elapsedMs,      setElapsedMs]      = useState(0)

  // Derived
  const selectedPreset = PRESETS[preset]
  const selectedGpu = GPU_OPTIONS.find(g => g.key === gpuKey) ?? GPU_OPTIONS[3]
  const effectiveResolution = advanced.resolution ?? selectedPreset.data.resolution
  const effectiveModelSize  = advanced.model_size_dims ?? selectedPreset.model.model_size_dims

  const estHours = useMemo(() => estimateRuntimeHours({
    sku:        selectedGpu.sku,
    pairs,
    resolution: effectiveResolution,
    maxSteps,
  }), [selectedGpu.sku, pairs, effectiveResolution, maxSteps])

  const hourlyRate = pricePerHour(selectedGpu.sku)
  const estCostUSD = estHours * hourlyRate

  function autoName() {
    if (name) return
    const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)
    setName(`tunet-${preset}-${stamp}`)
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (submitting) return
    setSubmitError(null)

    if (!name.trim()) { setSubmitError('Job name is required'); return }

    // Either we have picked files OR manual ShareSync paths
    const usingPickedFiles = !!picked && picked.src.length > 0 && picked.dst.length > 0
    if (!usingPickedFiles && (!srcDir || !dstDir)) {
      setSubmitError('Pick a project folder or enter ShareSync paths'); return
    }

    if (selectedGpu.badge === 'expensive') {
      const msg = `${selectedGpu.label} costs about $${hourlyRate.toFixed(2)}/hr (~$${estCostUSD.toFixed(0)} for this run).\n\nProceed?`
      if (!window.confirm(msg)) return
    }

    setSubmitting(true)
    setProgressEvents([])
    setProgressError(null)
    setProgressOpen(true)
    setProgressStartTs(Date.now())
    setElapsedMs(0)

    let stageId: string | null = null

    // ── 1. Upload picker files to /api/spark/upload-stage in batches ────────
    if (usingPickedFiles && picked) {
      try {
        stageId = await uploadStage(picked, (sent, total, batchIdx, totalBatches) => {
          setProgressEvents(prev => [...prev, {
            phase:      'stage',
            status:     sent === total ? 'done' : 'progress',
            sentBytes:  sent,
            totalBytes: total,
            files:      batchIdx,           // reuse 'files' for batch counter
            ms:         totalBatches,       // reuse 'ms' for total batch count (hacky but ok)
          }])
        })
        setProgressEvents(prev => [...prev, { phase: 'stage', status: 'done',
          sentBytes: picked.totalBytes, totalBytes: picked.totalBytes }])
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Upload failed'
        setProgressError(msg); setSubmitError(msg); setSubmitting(false); return
      }
    }

    const body = JSON.stringify({
      name:    name.trim(),
      preset,
      gpu:     gpuKey,
      ...(stageId ? { stageId } : {}),
      inputs: {
        // Only relevant when not using a stage
        src_dir:     srcDir,
        dst_dir:     dstDir,
        val_src_dir: valSrcDir.trim() || null,
        val_dst_dir: valDstDir.trim() || null,
        output_dir:  `/output/${name.trim().replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64) || 'tunet'}`,
      },
      advanced: { ...advanced, max_steps: maxSteps },
      idleHoldSeconds: idleHold,
    })

    try {
      const res = await fetch('/api/spark/training-jobs', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })

      // Validation errors come back as plain JSON 4xx
      if (!res.ok) {
        const j = await res.json().catch(() => ({}))
        throw new Error(j.error ?? `HTTP ${res.status}`)
      }
      if (!res.body) throw new Error('No response stream')

      // Consume SSE events
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let resolvedJobId: string | null = null
      let streamError: string | null = null

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        // Split on SSE event boundaries (\n\n)
        let idx
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx)
          buf = buf.slice(idx + 2)

          // Parse "data: {...}" lines (skip ":keepalive" comments)
          const dataLine = block.split('\n').find(l => l.startsWith('data: '))
          if (!dataLine) continue
          let evt: SubmitEvent
          try {
            evt = JSON.parse(dataLine.slice(6)) as SubmitEvent
          } catch { continue }

          setProgressEvents(prev => [...prev, evt])

          if (evt.phase === 'done')  resolvedJobId = evt.jobId ?? null
          if (evt.phase === 'error') streamError   = evt.error ?? 'Submit failed'
        }
      }

      if (streamError) throw new Error(streamError)
      if (!resolvedJobId) throw new Error('Submit completed but no jobId returned')

      // Brief moment so the user sees the green checkmark, then redirect
      await new Promise(resolve => setTimeout(resolve, 700))
      router.push(`/demo/jobs/${resolvedJobId}`)
    } catch (e) {
      setProgressError(e instanceof Error ? e.message : 'Submit failed')
      setSubmitError(e instanceof Error ? e.message : 'Submit failed')
      setSubmitting(false)
    }
  }

  // Tick elapsed time while the progress overlay is open
  useEffect(() => {
    if (!progressOpen || progressError) return
    const id = window.setInterval(() => {
      setElapsedMs(Date.now() - progressStartTs)
    }, 100)
    return () => window.clearInterval(id)
  }, [progressOpen, progressError, progressStartTs])

  return (
    <>
    <SubmitProgress
      open={progressOpen}
      events={progressEvents}
      errored={progressError}
      elapsedMs={elapsedMs}
      onClose={() => {
        setProgressOpen(false)
        setSubmitting(false)
      }}
      onRetry={() => {
        setProgressEvents([])
        setProgressError(null)
        setProgressStartTs(Date.now())
        setElapsedMs(0)
        // Re-trigger submit by simulating form submit
        const form = document.querySelector('form')
        if (form) form.requestSubmit()
      }}
    />
    <div className="space-y-5 animate-slide-in max-w-3xl">
      <div>
        <Link href="/demo/jobs" className="text-sm text-[#6b7280] hover:text-[#374151]">← All Jobs</Link>
        <h1 className="text-2xl font-bold text-[#111827] mt-2">New Training Job</h1>
        <p className="text-sm text-[#6b7280] mt-1">
          Pick a preset, point at frames, and go. The full tunet source bundle (~760 KB) is packed and shipped on submit.
        </p>
      </div>

      <form onSubmit={onSubmit} className="space-y-5">
        {/* ── Step 1: Name ──────────────────────────────────── */}
        <Card step={1} title="Job Name">
          <FormRow label="Name" hint="A friendly name for this run" required>
            <div className="flex gap-2">
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. tunet-beauty-shot-042"
                onFocus={autoName}
                required
              />
              <button
                type="button"
                onClick={() => { setName(''); autoName() }}
                className="px-3 py-2 text-xs text-[#6b7280] border border-[#e5e7eb] rounded-lg hover:bg-[#F9FAFB] whitespace-nowrap"
              >
                Auto-name
              </button>
            </div>
          </FormRow>
        </Card>

        {/* ── Step 2: Data ──────────────────────────────────── */}
        <Card step={2} title="Training Data">
          <p className="text-xs text-[#6b7280] mb-4">
            Pick your project folder to auto-detect <code className="bg-[#F9FAFB] px-1 rounded">src/</code>, <code className="bg-[#F9FAFB] px-1 rounded">dst/</code>, and validation folders.
            Or enter ShareSync paths manually below.
          </p>

          <FolderPicker
            onPicked={(p) => {
              setPicked(p)
              // Auto-fill ShareSync paths using the *actual* folder names
              // detected (e.g. 'input' → '/Compute Inputs/<root>/input'). Users
              // can edit if their ShareSync layout differs.
              const srcFolder = p.folderNames.src ?? 'src'
              const dstFolder = p.folderNames.dst ?? 'dst'
              if (p.src.length > 0) setSrcDir(`/Compute Inputs/${p.rootName}/${srcFolder}`)
              if (p.dst.length > 0) setDstDir(`/Compute Inputs/${p.rootName}/${dstFolder}`)
              if (p.valSrc.length > 0) setValSrcDir(`/Compute Inputs/${p.rootName}/${p.folderNames.val_src}`)
              else                     setValSrcDir('')
              if (p.valDst.length > 0) setValDstDir(`/Compute Inputs/${p.rootName}/${p.folderNames.val_dst}`)
              else                     setValDstDir('')
              if (p.pairCount > 0) {
                setPairs(p.pairCount)
                setPairsAutoDetected(true)
              }
              // If they picked a folder, surface manual-edit by default
              setShowManualPaths(true)
            }}
          />

          {picked && (
            <p className="mt-3 text-xs text-[#6b7280]">
              <strong className="text-[#16A34A]">Detected {picked.pairCount} pairs.</strong>{' '}
              You&apos;ll need to upload these frames to ShareSync (e.g. via Finder/WebDAV)
              before training can read them. The paths below assume <code className="bg-[#F9FAFB] px-1 rounded">/Compute Inputs/{picked.rootName}/...</code>;
              edit them if your actual ShareSync layout differs.
            </p>
          )}

          {/* Manual paths — collapsible */}
          <div className="mt-4">
            <button
              type="button"
              onClick={() => setShowManualPaths(!showManualPaths)}
              className="text-xs font-semibold text-[#7E3AF2] hover:text-[#6C2BD9] flex items-center gap-1"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                style={{ transform: showManualPaths ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 150ms' }}>
                <polyline points="9 18 15 12 9 6" />
              </svg>
              {showManualPaths ? 'Edit ShareSync paths' : 'Or enter ShareSync paths manually'}
            </button>

            {showManualPaths && (
              <div className="mt-3 space-y-3 p-4 bg-[#F9FAFB] border border-[#e5e7eb] rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FormRow label="Source frames (ShareSync path)" required>
                    <Input
                      value={srcDir}
                      onChange={(e) => setSrcDir(e.target.value)}
                      placeholder="/Compute Inputs/<project>/src"
                      className="font-mono text-xs"
                      required
                    />
                  </FormRow>
                  <FormRow label="Target frames (ShareSync path)" required>
                    <Input
                      value={dstDir}
                      onChange={(e) => setDstDir(e.target.value)}
                      placeholder="/Compute Inputs/<project>/dst"
                      className="font-mono text-xs"
                      required
                    />
                  </FormRow>
                  <FormRow label="Validation src (optional)">
                    <Input
                      value={valSrcDir}
                      onChange={(e) => setValSrcDir(e.target.value)}
                      placeholder=""
                      className="font-mono text-xs"
                    />
                  </FormRow>
                  <FormRow label="Validation dst (optional)">
                    <Input
                      value={valDstDir}
                      onChange={(e) => setValDstDir(e.target.value)}
                      placeholder=""
                      className="font-mono text-xs"
                    />
                  </FormRow>
                </div>

                <FormRow
                  label={pairsAutoDetected ? 'Frame count (auto-detected)' : 'Frame count (estimate)'}
                  hint={pairsAutoDetected
                    ? 'Detected from folder pick. Edit only if different from what gets uploaded to ShareSync.'
                    : 'How many src/dst pairs you have. Used only to estimate cost.'}
                >
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      min={1}
                      value={pairs}
                      onChange={(e) => {
                        setPairs(parseInt(e.target.value || '1', 10))
                        setPairsAutoDetected(false)
                      }}
                      className="w-32"
                    />
                    {pairsAutoDetected && (
                      <span className="text-[10px] uppercase tracking-wider text-[#16A34A] font-bold">auto</span>
                    )}
                  </div>
                </FormRow>
              </div>
            )}
          </div>
        </Card>

        {/* ── Step 3: Preset ────────────────────────────────── */}
        <Card step={3} title="Task Preset">
          <p className="text-xs text-[#6b7280] mb-4">
            Pick the preset that matches your task. Each one auto-configures model, loss, and training settings.
            You can override anything in Advanced below.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.values(PRESETS).map(p => (
              <PresetCard
                key={p.key}
                preset={p}
                selected={p.key === preset}
                onClick={() => setPreset(p.key)}
              />
            ))}
          </div>

          {/* Recommended (read-only) */}
          <div className="mt-5">
            <p className="text-xs font-semibold text-[#374151] mb-2 uppercase tracking-wider">
              Auto-configured
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <ReadonlyField label="Model"      value={`${selectedPreset.model.model_type.toUpperCase()} ${effectiveModelSize}`} />
              <ReadonlyField label="Resolution" value={`${effectiveResolution}px`} />
              <ReadonlyField label="Loss"       value={selectedPreset.training.loss} />
              <ReadonlyField label="Auto-Mask"  value={selectedPreset.mask.use_auto_mask ? 'enabled' : 'off'} />
            </div>
          </div>

          {/* Advanced overrides (collapsible) */}
          <div className="mt-4">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs font-semibold text-[#7E3AF2] hover:text-[#6C2BD9] flex items-center gap-1"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                style={{ transform: showAdvanced ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 150ms' }}>
                <polyline points="9 18 15 12 9 6" />
              </svg>
              Advanced Settings
            </button>

            {showAdvanced && (
              <div className="mt-3 p-4 bg-[#F9FAFB] border border-[#e5e7eb] rounded-lg space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FormRow label="Resolution">
                    <select
                      value={advanced.resolution ?? selectedPreset.data.resolution}
                      onChange={(e) => setAdvanced({ ...advanced, resolution: parseInt(e.target.value, 10) })}
                      className="w-full border border-[#e5e7eb] rounded-lg px-3.5 py-2.5 text-sm bg-white"
                    >
                      {RESOLUTIONS.map(r => <option key={r} value={r}>{r}px</option>)}
                    </select>
                  </FormRow>
                  <FormRow label="Model Size">
                    <select
                      value={advanced.model_size_dims ?? selectedPreset.model.model_size_dims}
                      onChange={(e) => setAdvanced({ ...advanced, model_size_dims: parseInt(e.target.value, 10) })}
                      className="w-full border border-[#e5e7eb] rounded-lg px-3.5 py-2.5 text-sm bg-white"
                    >
                      {MODEL_SIZES.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </FormRow>
                  <FormRow label="Batch Size">
                    <select
                      value={advanced.batch_size ?? 2}
                      onChange={(e) => setAdvanced({ ...advanced, batch_size: parseInt(e.target.value, 10) })}
                      className="w-full border border-[#e5e7eb] rounded-lg px-3.5 py-2.5 text-sm bg-white"
                    >
                      {BATCH_SIZES.map(b => <option key={b} value={b}>{b}</option>)}
                    </select>
                  </FormRow>
                  <FormRow label="Learning Rate">
                    <Input
                      value={advanced.lr ?? selectedPreset.training.lr}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value)
                        if (!Number.isNaN(v)) setAdvanced({ ...advanced, lr: v })
                      }}
                      type="number"
                      step="0.00001"
                    />
                  </FormRow>
                  <FormRow label="Loss Function">
                    <select
                      value={advanced.loss ?? selectedPreset.training.loss}
                      onChange={(e) => setAdvanced({ ...advanced, loss: e.target.value as AdvancedOverrides['loss'] })}
                      className="w-full border border-[#e5e7eb] rounded-lg px-3.5 py-2.5 text-sm bg-white"
                    >
                      {LOSSES.map(l => <option key={l.value} value={l.value}>{l.label}</option>)}
                    </select>
                  </FormRow>
                  <FormRow label="Max Steps" hint="0 = run until you stop it">
                    <Input
                      type="number"
                      min={0}
                      value={maxSteps}
                      onChange={(e) => setMaxSteps(parseInt(e.target.value || '0', 10))}
                    />
                  </FormRow>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* ── Step 4: Review & Train ─────────────────────────── */}
        <Card step={4} title="Review & Train">
          <FormRow label="GPU">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {GPU_OPTIONS.map(g => (
                <GpuChip
                  key={g.key}
                  gpu={g}
                  selected={g.key === gpuKey}
                  onClick={() => setGpuKey(g.key)}
                />
              ))}
            </div>
          </FormRow>

          <FormRow label="Idle hold (seconds)" hint="0 = stop instance immediately on exit. 300+ keeps warm for resubmits.">
            <Input
              type="number"
              min={0}
              max={3600}
              value={idleHold}
              onChange={(e) => setIdleHold(parseInt(e.target.value || '0', 10))}
              className="w-32"
            />
          </FormRow>

          {/* Cost estimate */}
          <div className="mt-4 bg-gradient-to-r from-[#F7F4FC] to-[#fdf4ff] border border-[#e9d5ff] rounded-lg p-4">
            <p className="text-xs font-semibold text-[#7E3AF2] uppercase tracking-wider mb-3">Estimate</p>
            <div className="space-y-1.5 text-sm">
              <CostRow label="GPU" value={`${selectedGpu.label} ${selectedGpu.vram} · $${hourlyRate.toFixed(2)}/hr`} />
              <CostRow label="Data" value={`${pairs} frame pairs · ${effectiveResolution}px`} />
              <CostRow label="Preset" value={`${selectedPreset.name} (${selectedPreset.model.model_type.toUpperCase()} ${effectiveModelSize})`} />
              <CostRow label="Steps" value={maxSteps > 0 ? `${maxSteps.toLocaleString()} max` : 'unlimited'} />
              <CostRow label="Estimated runtime" value={estHours < 1 ? `${(estHours * 60).toFixed(0)} minutes` : `${estHours.toFixed(1)} hours`} />
              <div className="pt-2 mt-2 border-t border-[#e9d5ff] flex items-center justify-between">
                <span className="font-semibold text-[#111827]">Estimated cost</span>
                <span className="font-bold text-lg text-[#7E3AF2]">~${estCostUSD.toFixed(2)}</span>
              </div>
            </div>
            <p className="text-[10px] text-[#9ca3af] mt-2">
              Estimate only — Spark bills per second on running. Actual cost depends on real T/Step which varies by content.
            </p>
          </div>

          {submitError && (
            <div className="mt-4 px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
              <strong>Submit failed:</strong> {submitError}
            </div>
          )}

          <div className="mt-5 flex items-center justify-between border-t border-[#e5e7eb] pt-5">
            <Link href="/demo/jobs" className="text-sm text-[#6b7280] hover:text-[#374151]">
              ← Cancel
            </Link>
            <Button type="submit" loading={submitting} disabled={submitting} size="lg">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              {submitting ? 'Submitting…' : `Start Training · ~$${estCostUSD.toFixed(2)}`}
            </Button>
          </div>
        </Card>
      </form>
    </div>
    </>
  )
}

// ── Subcomponents ───────────────────────────────────────────────────────────

function Card({ step, title, children }: { step: number; title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-5 card-shadow">
      <div className="flex items-center gap-2.5 mb-4">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-[#F7F4FC] text-[#7E3AF2] text-sm font-bold">
          {step}
        </span>
        <h2 className="text-base font-semibold text-[#111827]">{title}</h2>
      </div>
      {children}
    </div>
  )
}

function PresetCard({ preset, selected, onClick }: { preset: Preset; selected: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`text-left p-4 rounded-lg border-2 transition-all ${
        selected
          ? 'border-[#ae69f4] bg-[#F7F4FC]'
          : 'border-[#e5e7eb] bg-white hover:border-[#D1D5DB]'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="font-semibold text-[#111827] text-sm">{preset.name}</span>
        {selected && (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="#ae69f4">
            <path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
          </svg>
        )}
      </div>
      <p className="text-xs text-[#6b7280] leading-relaxed">{preset.description}</p>
      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {preset.tags.map(t => (
          <span key={t} className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-[#F9FAFB] text-[#6b7280] border border-[#e5e7eb]">
            {t}
          </span>
        ))}
      </div>
    </button>
  )
}

function GpuChip({ gpu, selected, onClick }: { gpu: GpuOption; selected: boolean; onClick: () => void }) {
  const badgeColor = (
    gpu.badge === 'recommended' ? 'text-[#ae69f4]' :
    gpu.badge === 'expensive'   ? 'text-[#D97706]' :
    gpu.badge === 'cheap'       ? 'text-[#16A34A]' :
    gpu.badge === 'fastest'     ? 'text-[#1c64f2]' :
                                  'text-[#9ca3af]'
  )
  const hourly = pricePerHour(gpu.sku)
  return (
    <button
      type="button"
      onClick={onClick}
      className={`text-left p-3 rounded-lg border-2 transition-all ${
        selected
          ? 'border-[#ae69f4] bg-[#F7F4FC]'
          : 'border-[#e5e7eb] bg-white hover:border-[#D1D5DB]'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="font-semibold text-sm text-[#111827]">{gpu.label}</span>
        <span className="text-xs text-[#6b7280]">{gpu.vram}</span>
      </div>
      <div className="mt-1 flex items-center justify-between">
        <span className={`text-[10px] uppercase tracking-wider font-bold ${badgeColor}`}>
          {gpu.badge ?? '·'}
        </span>
        <span className="text-xs text-[#374151] font-mono">${hourly.toFixed(2)}/hr</span>
      </div>
    </button>
  )
}

function ReadonlyField({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#F9FAFB] border border-[#e5e7eb] rounded px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">{label}</p>
      <p className="text-xs text-[#111827] font-mono mt-0.5">{value}</p>
    </div>
  )
}

// — Helper from the cost-box pattern in static HTML
function CostRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-[#6b7280]">{label}</span>
      <span className="text-[#374151]">{value}</span>
    </div>
  )
}

// Re-export so server can call without importing 'server-only' chain
function pricePerHourReExport() { return pricePerHour }   // tree-shake hint
void pricePerHourReExport

// ── Stage upload ────────────────────────────────────────────────────────────

const MAX_BATCH_BYTES = 200 * 1024 * 1024  // 200 MB per batch — keeps memory low

/**
 * Upload picker files in batches keyed by role. Files larger than
 * MAX_BATCH_BYTES are sent one per request; smaller files are grouped to
 * reduce request overhead. Reports total-bytes-sent progress to onProgress.
 *
 * Returns the stageId for the upload-stage server.
 */
async function uploadStage(
  picked: FolderPickerResult,
  onProgress: (sent: number, total: number, batchIdx: number, totalBatches: number) => void,
): Promise<string> {
  const batches: { role: string; files: File[]; bytes: number }[] = []

  function addRole(role: string, entries: { name: string; size: number; file: File }[]) {
    if (entries.length === 0) return
    let cur: File[] = []
    let curBytes = 0
    for (const e of entries) {
      if (curBytes + e.size > MAX_BATCH_BYTES && cur.length > 0) {
        batches.push({ role, files: cur, bytes: curBytes })
        cur = []
        curBytes = 0
      }
      cur.push(e.file)
      curBytes += e.size
    }
    if (cur.length > 0) batches.push({ role, files: cur, bytes: curBytes })
  }

  addRole('src',     picked.src)
  addRole('dst',     picked.dst)
  addRole('val_src', picked.valSrc)
  addRole('val_dst', picked.valDst)
  addRole('mask',    picked.mask)

  const total = batches.reduce((s, b) => s + b.bytes, 0)
  let sent = 0
  let stageId: string | null = null

  for (let i = 0; i < batches.length; i++) {
    const b = batches[i]
    const fd = new FormData()
    fd.set('role', b.role)
    if (stageId) fd.set('stageId', stageId)
    for (const f of b.files) fd.append('files', f, f.name)

    const res = await fetch('/api/spark/upload-stage', { method: 'POST', body: fd })
    if (!res.ok) {
      const j = await res.json().catch(() => ({}))
      throw new Error(j.error ?? `upload-stage HTTP ${res.status}`)
    }
    const json = await res.json() as { stageId: string }
    stageId = json.stageId
    sent += b.bytes
    onProgress(sent, total, i + 1, batches.length)
  }

  if (!stageId) throw new Error('No batches uploaded — picker had no files')
  return stageId
}
