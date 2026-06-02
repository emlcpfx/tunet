'use client'

/**
 * /jobs/new — preset-driven training job submitter.
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

import { Suspense, useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useSession } from 'next-auth/react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input, FormRow, InfoTip } from '@/components/ui/input'
import { FolderPicker, type FolderPickerResult } from '@/components/spark/folder-picker'
import { SubmitProgress, type SubmitEvent } from '@/components/spark/submit-progress'
import { PreviewFilterDialog } from '@/components/spark/preview-filter-dialog'
import { PairReviewDialog } from '@/components/spark/pair-review-dialog'
import { AutoMaskDialog } from '@/components/spark/auto-mask-dialog'
import { AdvancedSettings } from '@/components/spark/advanced-settings'
import { ModePicker } from '@/components/spark/mode-picker'
import { SourceJobPicker } from '@/components/spark/source-job-picker'
import { YamlConfigImporter } from '@/components/spark/yaml-config-importer'
import {
  PRESETS, type Preset, type PresetKey, type AdvancedOverrides,
  estimateTraining, pricePerHour, resolveColorSpace, DEFAULT_MAX_STEPS,
} from '@/lib/spark-presets'
import {
  buildFormState, parseFormState,
  type TrainingMode, type SourceJobRef, type ComputeMode,
} from '@/lib/spark-form-state'
import { uploadStage } from '@/lib/upload-stage'
import { jobLabel } from '@/lib/spark-types'
import type { SparkJob } from '@/lib/spark-types'
import type { PricingMap } from '@/app/api/spark/pricing/route'

interface GpuOption {
  key:   string
  sku:   string
  label: string
  vram:  string
  badge?: 'recommended' | 'cheap' | 'fastest' | 'expensive'
}

// Multi-GPU options removed: spark_start.sh:71 runs `python train.py` (not
// `torchrun`), so DDP never engages — multi-GPU SKUs would just charge 4×/8×
// for the same wall-clock as 1×. Re-add once we wire torchrun + benchmark
// the actual scaling factor.
// Order: cheapest → fastest. Badges reflect 2026-05-01 calibration:
//   - T4   = cheapest hourly, but slow → only wins for tiny test runs
//   - L4   = best $/k-step among the cheap tier; sweet spot for low budget
//   - A10  = ~3× faster than T4 at 2.2× the price → best $-vs-time balance
//   - L40S = no longer recommended; current eligible host (g6e.8xlarge) is
//            +67% $/hr without a proportional speedup → ~3× $/k-step of A10
//   - RTX PRO 6000 = absolute fastest, also pricey but cheaper per-step than L40S
// See benchmark.md for the underlying numbers.
const GPU_OPTIONS: GpuOption[] = [
  { key: 't4',         sku: 'g4dn.xlarge', label: 'T4',           vram: '16GB', badge: 'cheap'       },
  { key: 'l4',         sku: 'g6.2xlarge',  label: 'L4',           vram: '24GB'                       },
  { key: 'a10',        sku: 'g5.xlarge',   label: 'A10',          vram: '24GB', badge: 'recommended' },
  { key: 'rtxpro6000', sku: 'g7e.2xlarge', label: 'RTX PRO 6000', vram: '96GB', badge: 'fastest'     },
  { key: 'l40s',       sku: 'g6e.8xlarge', label: 'L40S',         vram: '48GB'                       },
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
  // useSearchParams requires a Suspense boundary in Next 15 — wrap the actual
  // form so the page can still SSR.
  return (
    <Suspense fallback={<div className="text-sm text-[#9ca3af] p-6">Loading…</div>}>
      <NewJobPageInner />
    </Suspense>
  )
}

function NewJobPageInner() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const cloneFromId = searchParams.get('clone')
  const resumeFromId = searchParams.get('resume')

  // Form state
  const [name,    setName]    = useState('')
  const [mode,    setMode]    = useState<TrainingMode>('new')
  // Source job for resume / fine-tune (null = unset). The SourceJobPicker
  // resolves it when the user picks a job + checkpoint.
  const [source,  setSource]  = useState<SourceJobRef | null>(null)
  const [srcDir,  setSrcDir]  = useState('/input/data/src')
  const [dstDir,  setDstDir]  = useState('/input/data/dst')
  const [valSrcDir, setValSrcDir] = useState<string>('')
  const [valDstDir, setValDstDir] = useState<string>('')
  const [preset,  setPreset]  = useState<PresetKey>('beauty')
  const [gpuKey,  setGpuKey]  = useState<string>('a10')
  const [advanced, setAdvanced] = useState<AdvancedOverrides>({})
  const [showAdvanced, setShowAdvanced] = useState(false)
  // Default to a 150k safety cap (DEFAULT_MAX_STEPS) instead of unlimited: an
  // un-capped run that can't be cancelled (e.g. a stale-session cancel) bills
  // indefinitely — see the 75h dollywood runs. 150k is well past a typical
  // converged run; the user can raise it or set 0 (no limit) below.
  const [maxSteps, setMaxSteps] = useState<number>(DEFAULT_MAX_STEPS)
  const [pairs, setPairs] = useState<number>(200)   // user-supplied or auto-detected from folder pick
  const [pairsAutoDetected, setPairsAutoDetected] = useState(false)
  const [idleHold, setIdleHold] = useState(0)
  // Compute mode — InstantCompute (guaranteed availability) vs
  // SmartCompute (preemptible, ~60% cheaper, auto-retries on interrupt).
  // Defaults to InstantCompute: TuNet's per-run setup cost is high (~5-8 min
  // for torch.compile warmup + auto-batch sizing + first-step compile), which
  // is longer than the time-to-first-checkpoint. Spot preemption inside that
  // window loses the whole setup and a retry just repeats it — observed three
  // back-to-back spot kills before any training step ran. Spot only pays off
  // on long runs that checkpoint early; flip to Smart in one click for those.
  const [computeMode, setComputeMode] = useState<ComputeMode>('instant')
  // SmartCompute retry budget. Default 2 (one above Spark's default 1). Range
  // [0, 5] per docs. Ignored when computeMode='instant'.
  const [maxRetriesOnInterrupt, setMaxRetriesOnInterrupt] = useState<number>(2)
  // Training-alert prefs — read by api/cron/training-alerts. Defaulting both
  // to ON because the whole point is to save the user money; opting out is
  // explicit. Email auto-fills from the signed-in Keycloak session below;
  // the user can still override or clear it to opt out per-job.
  const [alertEmail,    setAlertEmail]    = useState('')
  const [alertPlateau,  setAlertPlateau]  = useState(true)
  const [alertDiverging, setAlertDiverging] = useState(true)
  const [alertSpot,     setAlertSpot]     = useState(true)
  // Whether the user has manually touched the email field. Once true, we
  // stop overwriting from the session — they may want to send alerts to a
  // shared address or clear the field to opt out.
  const [alertEmailDirty, setAlertEmailDirty] = useState(false)

  // Auto-populate the alerts email from the signed-in session. Only fires
  // when the field is empty and the user hasn't manually typed in it — won't
  // clobber a clone-rehydrated value or anything the user has set by hand.
  const { data: session } = useSession()
  useEffect(() => {
    const sessionEmail = session?.user?.email
    if (!alertEmailDirty && !alertEmail && typeof sessionEmail === 'string' && sessionEmail) {
      setAlertEmail(sessionEmail)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally
    // only re-runs when the session email arrives or the user clears the
    // dirty flag (currently never); not when alertEmail itself changes.
  }, [session?.user?.email])
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

  // Preview Filter dialog runs entirely in the browser via a Web Worker now —
  // no upload needed. We just compute matched src↔dst File pairs from the
  // picked folder and hand them to the dialog when it opens.
  const [filterOpen, setFilterOpen]       = useState(false)
  const [filterError, setFilterError]     = useState<string | null>(null)

  // Pair Review — a hard gate before submit. The user must visually approve the
  // matched src↔dst pairs (Source | Target | Diff) so a wrong or misaligned
  // dataset can't reach a paid training run. Approval is reset whenever the
  // picked dataset changes (effect on filterPairs below), forcing a fresh look.
  const [reviewOpen, setReviewOpen]         = useState(false)
  const [reviewApproved, setReviewApproved] = useState(false)
  // Skip threshold lives on `advanced.skip_empty_threshold` so the Preview
  // Filter dialog and the Mask & Skip-empty section in Advanced Settings
  // share one value. Default 3.0 matches the desktop app + base/base.yaml.
  const skipThreshold = advanced.skip_empty_threshold ?? 3.0
  const setSkipThreshold = (t: number) =>
    setAdvanced({ ...advanced, skip_empty_threshold: t })

  // Auto-mask gamma — same pattern as skipThreshold. Lives on
  // advanced.auto_mask_gamma so the dialog and the form stay in sync.
  const [autoMaskOpen, setAutoMaskOpen] = useState(false)
  const autoMaskGamma =
    advanced.auto_mask_gamma ?? PRESETS[preset].mask.auto_mask_gamma ?? 1.0
  const setAutoMaskGamma = (g: number) =>
    setAdvanced({ ...advanced, auto_mask_gamma: g })
  const openAutoMaskPreview = () => {
    if (!filterPairs || filterPairs.length === 0) return
    setAutoMaskOpen(true)
  }

  // Match src↔dst File handles by stem. Stable across renders so the dialog's
  // scan-cache effect can detect "same dataset" and not redo work.
  const filterPairs = useMemo(() => {
    if (!picked || picked.src.length === 0 || picked.dst.length === 0) return null
    const dstByStem = new Map<string, File>()
    for (const e of picked.dst) dstByStem.set(stemLower(e.name), e.file)
    const out: { name: string; src: File; dst: File }[] = []
    for (const e of picked.src) {
      const m = dstByStem.get(stemLower(e.name))
      if (m) out.push({ name: e.name, src: e.file, dst: m })
    }
    return out
  }, [picked])

  // Whether there's a local dataset to review/gate on. Manual ShareSync paths
  // have no File handles to diff, so they're not gated (same limitation as the
  // Preview Filter / Auto-Mask previews).
  const hasReviewablePairs = !!filterPairs && filterPairs.length > 0

  // Re-arm the gate whenever the picked dataset changes — filterPairs identity
  // only flips when `picked` changes, so this won't fire on unrelated renders.
  useEffect(() => { setReviewApproved(false) }, [filterPairs])

  function openPreviewFilter() {
    if (!filterPairs || filterPairs.length === 0) {
      setFilterError('Pick a project folder with matching src/ and dst/ files first.')
      setFilterOpen(true)
      return
    }
    setFilterError(null)
    setFilterOpen(true)
  }

  // ── Clone-from-job rehydration ─────────────────────────────────────────────
  // When ?clone=<jobId> is present, fetch that job and rehydrate the form
  // from its TUNET_FORM_STATE env stash. Falls back to inferring from
  // env.TUNET_PRESET / env.TUNET_GPU when the stash isn't present (older jobs).
  // The dataset (folder pick) is intentionally NOT cloned — the user has to
  // pick frames again so we don't silently re-use a different project.
  const [cloneSource, setCloneSource] = useState<{ id: string; label: string } | null>(null)
  const [cloneError, setCloneError] = useState<string | null>(null)
  useEffect(() => {
    if (!cloneFromId) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch(`/api/spark/jobs/${cloneFromId}`, { cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const job = data.job as SparkJob | undefined
        if (!job)             throw new Error('source job not found')
        if (cancelled)        return
        applyClonedJob(job)
      } catch (e) {
        if (!cancelled) setCloneError(e instanceof Error ? e.message : 'Clone failed')
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional one-shot on mount
  }, [cloneFromId])

  // ── Resume-from-job ────────────────────────────────────────────────────────
  // ?resume=<jobId> is a one-click entry point from a failed/finished job's
  // detail page. It rehydrates the full form state (so preset/model/loss match
  // — train.py refuses to load a checkpoint with mismatched architecture), then
  // flips to resume mode and pre-selects the source job. SourceJobPicker reads
  // source.jobId on mount and auto-loads the job's latest checkpoint.
  const [resumeError, setResumeError] = useState<string | null>(null)
  useEffect(() => {
    if (!resumeFromId) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch(`/api/spark/jobs/${resumeFromId}`, { cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const job = data.job as SparkJob | undefined
        if (!job)      throw new Error('source job not found')
        if (cancelled) return
        applyClonedJob(job)
        setMode('resume')
        setSource({ jobId: job.id, jobLabel: jobLabel(job), checkpointName: '' })
      } catch (e) {
        if (!cancelled) setResumeError(e instanceof Error ? e.message : 'Resume failed')
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional one-shot on mount
  }, [resumeFromId])

  function applyClonedJob(job: SparkJob) {
    setCloneSource({ id: job.id, label: jobLabel(job) })

    const stashed = parseFormState(job.env?.TUNET_FORM_STATE)
    if (stashed) {
      // Full rehydration from the stash — most accurate.
      setPreset(stashed.preset in PRESETS ? stashed.preset : 'beauty')
      setGpuKey(stashed.gpuKey)
      setComputeMode(stashed.computeMode === 'smart' ? 'smart' : 'instant')
      setMaxRetriesOnInterrupt(
        Math.max(0, Math.min(5, Math.floor(stashed.maxRetriesOnInterrupt ?? 2))),
      )
      // Merge skipThreshold into advanced.skip_empty_threshold in one shot —
      // calling setSkipThreshold separately would race with setAdvanced
      // because both spread the same closure-captured `advanced`.
      setAdvanced({
        ...(stashed.advanced ?? {}),
        skip_empty_threshold: stashed.skipThreshold,
      })
      setMaxSteps(stashed.maxSteps)
      setIdleHold(stashed.idleHoldSeconds)
      if (stashed.pairs > 0) {
        setPairs(stashed.pairs)
        setPairsAutoDetected(false)
      }
      setAlertEmail(stashed.alerts.email)
      setAlertPlateau(stashed.alerts.plateau)
      setAlertDiverging(stashed.alerts.diverging)
      setAlertSpot(stashed.alerts.spot)
      if (stashed.manual) {
        setShowManualPaths(true)
        setSrcDir(stashed.manual.srcDir)
        setDstDir(stashed.manual.dstDir)
        setValSrcDir(stashed.manual.valSrcDir)
        setValDstDir(stashed.manual.valDstDir)
      }
    } else {
      // Older job without a form-state stash — fall back to env hints.
      const presetGuess = job.env?.TUNET_PRESET
      if (presetGuess && presetGuess in PRESETS) setPreset(presetGuess as PresetKey)
      const gpuGuess = job.env?.TUNET_GPU
      if (gpuGuess) setGpuKey(gpuGuess)
      const emailGuess = job.env?.TUNET_ALERT_EMAIL
      if (emailGuess) {
        setAlertEmail(emailGuess)
        setAlertPlateau(job.env?.TUNET_ALERT_PLATEAU === '1')
        setAlertDiverging(job.env?.TUNET_ALERT_DIVERGING === '1')
        setAlertSpot(job.env?.TUNET_ALERT_SPOT === '1')
      }
    }

    // Suggest a derived name like "<prior> v2" so the user is nudged to rename
    // and can't accidentally clobber the cloned job's output dir.
    const base = jobLabel(job).replace(/\s+v\d+$/i, '').trim()
    setName(base ? `${base} v2` : '')
  }

  // Derived
  const selectedPreset = PRESETS[preset]
  // Fallback to A10 (the 2026-05-01 recommended pick) if the saved gpuKey
  // doesn't match any current option (e.g. cloned from an old multi-GPU job).
  const selectedGpu = GPU_OPTIONS.find(g => g.key === gpuKey) ?? GPU_OPTIONS.find(g => g.key === 'a10') ?? GPU_OPTIONS[0]
  const effectiveResolution = advanced.resolution ?? selectedPreset.data.resolution
  const effectiveModelSize  = advanced.model_size_dims ?? selectedPreset.model.model_size_dims

  // Training-time estimate. Returns a range + basis so we can display low–high
  // and "why this estimate?" rather than a fake-precise single number. Pulls
  // in the heavy-impact settings (model size, type, loss, batch) — see
  // estimateTraining() in spark-presets.ts.
  const estimate = useMemo(() => estimateTraining({
    sku:               selectedGpu.sku,
    pairs,
    resolution:        effectiveResolution,
    maxSteps,
    model_size_dims:   effectiveModelSize,
    model_type:        advanced.model_type ?? selectedPreset.model.model_type,
    loss:              advanced.loss       ?? selectedPreset.training.loss,
    batch_size:        advanced.batch_size,
  }), [
    selectedGpu.sku, pairs, effectiveResolution, maxSteps, effectiveModelSize,
    advanced.model_type, advanced.loss, advanced.batch_size,
    selectedPreset.model.model_type, selectedPreset.training.loss,
  ])

  // Live pricing from POST /api/compute/jobs/estimate, cached server-side for
  // 5 min. Falls back to the hardcoded GPU_PRICING_USD_PER_HR table only if
  // the fetch hasn't landed yet or Spark's estimate endpoint is unreachable.
  const [livePricing, setLivePricing] = useState<PricingMap | null>(null)
  useEffect(() => {
    fetch('/api/spark/pricing', { cache: 'no-store' })
      .then(r => (r.ok ? r.json() : null))
      .then((data: PricingMap | null) => { if (data) setLivePricing(data) })
      .catch(() => { /* fall back to hardcoded prices */ })
  }, [])
  // Look up the live rate for the user's selected compute mode. Smart-mode
  // rates can fall back to instant if Spark hasn't yet observed smart pricing
  // for the SKU (the estimate endpoint flags that case in `notes[]`). If the
  // /api/spark/pricing call hasn't landed at all, fall back to the hardcoded
  // table — which is instant-only, so a smart-mode selection will briefly
  // show the instant rate until the live fetch completes.
  const priceFor = (sku: string): number => {
    const live = livePricing?.[sku]
    if (live) {
      if (computeMode === 'smart') {
        return live.smartUsdPerHr ?? live.instantUsdPerHr ?? pricePerHour(sku)
      }
      return live.instantUsdPerHr ?? pricePerHour(sku)
    }
    return pricePerHour(sku)
  }

  const hourlyRate     = priceFor(selectedGpu.sku)
  const estCostLowUSD  = estimate.lowHours  * hourlyRate
  const estCostHighUSD = estimate.highHours * hourlyRate

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

    // Resume / fine-tune require a source job + checkpoint
    if ((mode === 'resume' || mode === 'finetune') && (!source || !source.checkpointName)) {
      setSubmitError(`${mode === 'resume' ? 'Resume' : 'Fine-tune'} requires picking a source job and checkpoint`)
      return
    }

    // Hard gate: a picked dataset must be visually reviewed and approved before
    // it can be submitted (catches wrong/misaligned folders, which the pair
    // count and Preview Filter don't). Manual ShareSync paths aren't gated —
    // there are no local files to diff. Re-opens the dialog so approval is one
    // click away rather than a dead end.
    if (usingPickedFiles && !reviewApproved) {
      setSubmitError('Review and approve your image pairs before submitting (Step 3 → Review pairs).')
      setReviewOpen(true)
      return
    }

    if (selectedGpu.badge === 'expensive') {
      const msg = `${selectedGpu.label} costs about $${hourlyRate.toFixed(2)}/hr ` +
        `(estimated $${estCostLowUSD.toFixed(0)}–$${estCostHighUSD.toFixed(0)} for this run).\n\nProceed?`
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

    const formState = buildFormState({
      mode,
      ...(source ? { source } : {}),
      preset,
      gpuKey,
      computeMode,
      maxRetriesOnInterrupt,
      advanced,
      maxSteps,
      idleHoldSeconds: idleHold,
      pairs,
      skipThreshold,
      alerts:  { email: alertEmail.trim(), plateau: alertPlateau, diverging: alertDiverging, spot: alertSpot },
      manual:  showManualPaths
        ? { srcDir, dstDir, valSrcDir, valDstDir }
        : undefined,
    })

    const body = JSON.stringify({
      name:    name.trim(),
      preset,
      gpu:     gpuKey,
      mode,
      ...(source && (mode === 'resume' || mode === 'finetune')
        ? {
            source: {
              jobId:          source.jobId,
              checkpointName: source.checkpointName,
              // Critical for local-upload resume: without this field the
              // server sees jobId='local-upload' and tries to call getJob()
              // against the Spark API, which 404s. The route uses
              // localCheckpointStageId to skip the Spark lookup and read
              // the .pth straight from the upload-stage tmpdir instead.
              ...(source.localCheckpointStageId
                ? { localCheckpointStageId: source.localCheckpointStageId }
                : {}),
            },
          }
        : {}),
      ...(stageId ? { stageId } : {}),
      inputs: {
        // Only relevant when not using a stage
        src_dir:     srcDir,
        dst_dir:     dstDir,
        val_src_dir: valSrcDir.trim() || null,
        val_dst_dir: valDstDir.trim() || null,
        output_dir:  `/output/${name.trim().replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64) || 'tunet'}`,
      },
      advanced: {
        ...advanced,
        max_steps:   maxSteps,
        // Resolve the stop-on-plateau toggle's default (on) explicitly so the
        // submitted job + stashed form state always record the real setting,
        // rather than relying on the server-side default for an untouched box.
        es_enabled:  advanced.es_enabled ?? true,
        es_stop:     advanced.es_stop ?? true,
        // Resolve 'auto' here so the server only ever sees a concrete value.
        // tunet's YAML parser doesn't know what 'auto' means.
        color_space: resolveColorSpace(advanced.color_space, picked?.sampleExt),
      },
      idleHoldSeconds: idleHold,
      computeMode,
      ...(computeMode === 'smart' ? { maxRetriesOnInterrupt } : {}),
      alerts: alertEmail.trim() ? {
        email:     alertEmail.trim(),
        plateau:   alertPlateau,
        diverging: alertDiverging,
        spot:      alertSpot,
      } : undefined,
      formState,
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
      router.push(`/jobs/${resolvedJobId}`)
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
    <PreviewFilterDialog
      open={filterOpen}
      onClose={() => setFilterOpen(false)}
      pairs={filterPairs}
      resolution={effectiveResolution}
      overlap={selectedPreset.data.overlap_factor ?? 0.25}
      initialThreshold={skipThreshold}
      onAccept={(t) => setSkipThreshold(t)}
    />
    <AutoMaskDialog
      open={autoMaskOpen}
      onClose={() => setAutoMaskOpen(false)}
      pairs={filterPairs}
      initialGamma={autoMaskGamma}
      onAccept={(g) => setAutoMaskGamma(g)}
    />
    <PairReviewDialog
      open={reviewOpen}
      onClose={() => setReviewOpen(false)}
      pairs={filterPairs}
      onApprove={() => { setReviewApproved(true); setSubmitError(null) }}
    />
    {filterError && filterOpen && (
      <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4" onClick={() => setFilterOpen(false)}>
        <div className="bg-white rounded-lg p-5 max-w-md" onClick={e => e.stopPropagation()}>
          <h3 className="text-sm font-semibold text-[#EF4444]">Preview Filter</h3>
          <p className="text-xs text-[#374151] mt-1">{filterError}</p>
          <button onClick={() => setFilterOpen(false)} className="mt-3 px-3 py-1 text-xs bg-[#7E3AF2] text-white rounded">OK</button>
        </div>
      </div>
    )}
    <div className="space-y-5 animate-slide-in max-w-3xl">
      <div>
        <Link href="/jobs" className="text-sm text-[#6b7280] hover:text-[#374151]">← All Jobs</Link>
        <h1 className="text-2xl font-bold text-[#111827] mt-2">New Training Job</h1>
        <p className="text-sm text-[#6b7280] mt-1">
          Pick a preset, point at frames, and go. The full tunet source bundle (~760 KB) is packed and shipped on submit.
        </p>
      </div>

      {cloneSource && (
        <div className="px-4 py-3 bg-[#F7F4FC] border border-[#e9d5ff] rounded-lg flex items-start justify-between gap-3">
          <div className="text-sm text-[#374151] min-w-0">
            <p>
              <span className="font-semibold">Cloned from</span>{' '}
              <Link href={`/jobs/${cloneSource.id}`} className="font-mono text-xs text-[#7E3AF2] hover:underline">
                {cloneSource.label}
              </Link>
            </p>
            <p className="text-xs text-[#6b7280] mt-1">
              Settings are pre-filled. Pick your project folder again and adjust as needed.
            </p>
          </div>
          <button
            type="button"
            onClick={() => { setCloneSource(null); router.replace('/jobs/new') }}
            className="text-xs text-[#6b7280] hover:text-[#374151] flex-shrink-0"
            aria-label="Clear clone"
          >
            ×
          </button>
        </div>
      )}

      {(cloneError || resumeError) && (
        <div className="px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
          Couldn&apos;t load source job: {cloneError ?? resumeError}
        </div>
      )}

      <form onSubmit={onSubmit} className="space-y-5">
        {/* ── Step 1: Mode ──────────────────────────────────── */}
        <Card step={1} title="Training Mode">
          <ModePicker
            value={mode}
            onChange={(m) => {
              setMode(m)
              // Switching back to 'new' clears any picked source so a stale
              // checkpoint can't sneak into the submit body.
              if (m === 'new') setSource(null)
            }}
          />
          {(mode === 'resume' || mode === 'finetune') && (
            <div className="mt-4">
              <SourceJobPicker
                mode={mode}
                value={source}
                onChange={(ref, meta) => {
                  setSource(ref)
                  // Resume requires the new run to use the source's preset/gpu
                  // (train.py rejects loss/size mismatches). Auto-snap them
                  // here so the user doesn't have to remember.
                  if (ref && mode === 'resume') {
                    if (meta.preset && meta.preset in PRESETS) setPreset(meta.preset as PresetKey)
                    if (meta.gpuKey && GPU_OPTIONS.some(g => g.key === meta.gpuKey)) {
                      setGpuKey(meta.gpuKey)
                    }
                  }
                }}
              />
            </div>
          )}
          {/* YAML config importer — pre-fills Advanced Settings + Max Steps
              from a tunet model.yaml the user has on disk. Always available
              regardless of mode; useful for porting local runs into the
              browser form without re-typing each value. */}
          <YamlConfigImporter
            onLoaded={(r) => {
              // Merge with existing overrides so partial YAMLs don't blow
              // away whatever the user already had set.
              setAdvanced(prev => ({ ...prev, ...r.advanced }))
              if (typeof r.maxSteps === 'number') setMaxSteps(r.maxSteps)
              // Auto-open Advanced so the user can see what was applied
              setShowAdvanced(true)
            }}
          />
        </Card>

        {/* ── Step 2: Name ──────────────────────────────────── */}
        <Card step={2} title="Job Name">
          <FormRow
            label="Name"
            hint="A friendly name for this run"
            tip="Used as the output folder name and as the friendly title in the dashboard.

Convention: <project>_<task>_<version>, e.g.  hero_beauty_v3
Letters, numbers, _ and - only — anything else is sanitized."
            required
          >
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

        {/* ── Step 3: Data ──────────────────────────────────── */}
        <Card
          step={3}
          title="Training Data"
          info={
            'Accepted subfolder names (any one per role):\n' +
            '• Source:  src, source, in, input\n' +
            '• Target:  dst, dest, destination, out, output, target\n' +
            '• Validation src:  val_src, val_source, val_input\n' +
            '• Validation dst:  val_dst, val_dest, val_target\n' +
            '• Mask (optional):  mask, masks, matte, mattes\n\n' +
            'Image files: PNG, JPG/JPEG, EXR, TIF/TIFF, BMP, WEBP.\n\n' +
            'Source and target filenames must match (e.g. frame_001.exr in both folders).'
          }
        >
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
            <>
              <p className="mt-3 text-xs text-[#6b7280]">
                <strong className="text-[#16A34A]">Detected {picked.pairCount} pairs.</strong>{' '}
                You&apos;ll need to upload these frames to ShareSync (e.g. via Finder/WebDAV)
                before training can read them. The paths below assume <code className="bg-[#F9FAFB] px-1 rounded">/Compute Inputs/{picked.rootName}/...</code>;
                edit them if your actual ShareSync layout differs.
              </p>
              <div className="mt-3 flex items-center gap-3 flex-wrap">
                <button
                  type="button"
                  onClick={() => setReviewOpen(true)}
                  className={`px-3 py-1.5 text-xs font-semibold rounded border ${
                    reviewApproved
                      ? 'border-[#16A34A] text-[#16A34A] bg-[#F0FDF4] hover:bg-[#dcfce7]'
                      : 'border-[#7E3AF2] text-white bg-[#7E3AF2] hover:bg-[#6C2BD9]'
                  }`}
                >
                  {reviewApproved ? '✓ Pairs reviewed' : 'Review pairs'}
                </button>
                <span className="text-[11px] text-[#6b7280]">
                  {reviewApproved
                    ? 'Approved. Click to review again.'
                    : 'Required before submit — eyeball every Source ↔ Target ↔ Diff to catch a wrong or misaligned dataset.'}
                </span>
              </div>
              <div className="mt-2 flex items-center gap-3">
                <button
                  type="button"
                  onClick={openPreviewFilter}
                  className="px-3 py-1.5 text-xs font-semibold border border-[#7E3AF2] text-[#7E3AF2] rounded hover:bg-[#faf5ff]"
                >
                  Preview Filter
                </button>
                <span className="text-[11px] text-[#6b7280]">
                  Visualize which patches will be kept vs skipped at the empty-patch threshold.
                </span>
              </div>
            </>
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

        {/* ── Step 4: Preset ────────────────────────────────── */}
        <Card step={4} title="Task Preset">
          <p className="text-xs text-[#6b7280] mb-4">
            Pick the preset that matches your task. Each one auto-configures model, loss, and training settings.
            You can override anything in Advanced below.
          </p>

          {mode === 'resume' && (
            <p className="mb-3 px-3 py-2 text-xs text-[#7E3AF2] bg-[#F7F4FC] border border-[#e9d5ff] rounded">
              Locked to the source job&apos;s preset — train.py refuses to load a checkpoint with mismatched loss / model size.
            </p>
          )}

          <div className={`grid grid-cols-1 md:grid-cols-2 gap-3 ${mode === 'resume' ? 'opacity-50 pointer-events-none' : ''}`}>
            {Object.values(PRESETS).map(p => (
              <PresetCard
                key={p.key}
                preset={p}
                selected={p.key === preset}
                onClick={() => setPreset(p.key)}
              />
            ))}
          </div>

          {/* Recommended — Resolution is a one-click override; the rest are
              read-only summary chips (drill into Advanced to change them). */}
          <div className="mt-5">
            <p className="text-xs font-semibold text-[#374151] mb-2 uppercase tracking-wider">
              Auto-configured
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <ReadonlyField label="Model"      value={`${selectedPreset.model.model_type.toUpperCase()} ${effectiveModelSize}`} />
              <ResolutionField
                value={effectiveResolution}
                presetDefault={selectedPreset.data.resolution}
                isOverride={advanced.resolution !== undefined && advanced.resolution !== selectedPreset.data.resolution}
                onChange={(r) => setAdvanced({
                  ...advanced,
                  // Clearing back to the preset default removes the override
                  // entirely so the chip un-highlights.
                  resolution: r === selectedPreset.data.resolution ? undefined : r,
                })}
              />
              <ReadonlyField label="Loss"       value={selectedPreset.training.loss} />
              <ReadonlyField label="Auto-Mask"  value={selectedPreset.mask.use_auto_mask ? 'enabled' : 'off'} />
            </div>
            {/* T4 + 512+ warning. T4 has 16GB VRAM — fine at 512 batch=1-2 in
                AMP, but the default batch_size is 4. We don't auto-shrink
                batch_size (that's an Advanced thing) so we just warn. */}
            {gpuKey === 't4' && effectiveResolution >= 512 && (
              <p className="mt-2 text-xs text-[#D97706]">
                ⚠ T4 16GB at {effectiveResolution}px may need batch_size 1-2 (set in Advanced).
                Consider 384px for faster training without VRAM headroom worries.
              </p>
            )}
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
              <div className="mt-3 space-y-3">
                <AdvancedSettings
                  value={advanced}
                  onChange={setAdvanced}
                  preset={selectedPreset}
                  sampleExt={picked?.sampleExt ?? null}
                  onPreviewFilter={openPreviewFilter}
                  previewFilterReady={!!filterPairs && filterPairs.length > 0}
                  onAutoMaskPreview={openAutoMaskPreview}
                  autoMaskPreviewReady={!!filterPairs && filterPairs.length > 0}
                />
                <p className="text-[11px] text-[#9ca3af] italic">
                  Empty fields fall back to the preset default. Reset by re-selecting a preset above.
                </p>
              </div>
            )}
          </div>
        </Card>

        {/* ── Step 5: Review & Train ─────────────────────────── */}
        <Card step={5} title="Review & Train">
          <FormRow
            label="Compute mode"
            hint={computeMode === 'smart'
              ? `SmartCompute: cheaper but may be reclaimed mid-run. Auto-retries up to ${maxRetriesOnInterrupt}×.`
              : 'InstantCompute: warm-pool GPU, starts in seconds, never interrupted.'}
            tip={`InstantCompute (default)
  Warm-pool GPU. Starts in seconds, never preempted, full hourly rate.
  Best for interactive runs and anything you don't want to babysit.

SmartCompute (~60% off)
  Preemptible spare capacity. ~3 min cold start. The platform may reclaim
  the GPU mid-run with a short warning — Spark auto-re-queues your job on
  fresh smart capacity up to your retry budget.

  Spark Fuse pays the partial-attempt time on every retry, not you — but
  if your container can't resume from /output/ checkpoints, each retry
  re-runs from scratch.`}
          >
            <div className="flex items-center gap-2">
              <ComputeModeToggle value={computeMode} onChange={setComputeMode} />
              {computeMode === 'smart' && (
                <label className="flex items-center gap-1.5 text-xs text-[#6b7280]">
                  Retry budget
                  <Input
                    type="number"
                    min={0}
                    max={5}
                    value={maxRetriesOnInterrupt}
                    onChange={(e) => {
                      const n = parseInt(e.target.value || '0', 10)
                      setMaxRetriesOnInterrupt(Math.max(0, Math.min(5, n)))
                    }}
                    className="w-16"
                  />
                </label>
              )}
            </div>
          </FormRow>

          <FormRow label="GPU">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {GPU_OPTIONS.map(g => (
                <GpuChip
                  key={g.key}
                  gpu={g}
                  priceUsdPerHr={priceFor(g.sku)}
                  computeMode={computeMode}
                  selected={g.key === gpuKey}
                  onClick={() => setGpuKey(g.key)}
                />
              ))}
            </div>
          </FormRow>

          <FormRow
            label="Idle hold (seconds)"
            hint={computeMode === 'smart'
              ? 'Ignored on SmartCompute (warm-pool is InstantCompute-only).'
              : '0 = stop instance immediately on exit. 300+ keeps warm for resubmits.'}
            tip="How long to keep the GPU instance warm after training exits.

  0    — Stop immediately. Cheapest. Re-submitting later cold-starts (~3-5 min).
  300  — 5 min hold. Good if you might re-submit with tweaks.
  3600 — 1 hour hold. Useful for iterating live.

You're billed for idle time — keep this minimal unless you know you'll resubmit.

Note: idle-hold is InstantCompute-only. SmartCompute jobs always release the
GPU on exit (no warm-pool affinity in smart mode)."
          >
            <Input
              type="number"
              min={0}
              max={3600}
              value={idleHold}
              onChange={(e) => setIdleHold(parseInt(e.target.value || '0', 10))}
              className={`w-32 ${computeMode === 'smart' ? 'opacity-50' : ''}`}
              disabled={computeMode === 'smart'}
            />
          </FormRow>

          {/* Auto-stop — the billing backstop: hard step cap + stop-on-plateau */}
          <FormRow
            label="Auto-stop"
            hint="Bounds how long a run can train so a forgotten or un-cancellable job can't bill indefinitely. Defaults stop a typical run after it has converged."
            tip={`Max steps — a hard stop point. The run saves a final checkpoint and exits at this many steps. Default ${DEFAULT_MAX_STEPS.toLocaleString()} (~16h on an A10) is well above a typical converged run (~96k for ~50 frames). Set 0 for no limit — only if you're actively watching the run.

Stop on plateau — ends the run automatically once the smoothed loss stops improving for ~30 epochs (a final checkpoint is saved first). Leave on unless you want to judge convergence yourself.`}
          >
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  min={0}
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(Math.max(0, parseInt(e.target.value || '0', 10) || 0))}
                  className="w-36"
                />
                <span className="text-xs text-[#9ca3af]">
                  {maxSteps > 0
                    ? 'max steps (0 = no limit)'
                    : 'no limit — bills until you stop it'}
                </span>
              </div>
              <label className="flex items-center gap-1.5 cursor-pointer text-xs">
                <input
                  type="checkbox"
                  checked={advanced.es_stop ?? true}
                  onChange={(e) =>
                    setAdvanced({ ...advanced, es_enabled: e.target.checked, es_stop: e.target.checked })
                  }
                  className="accent-[#7E3AF2]"
                />
                <span className="text-[#374151]">
                  Stop automatically when training plateaus
                  <span className="text-[#9ca3af]"> (saves a final checkpoint; recommended)</span>
                </span>
              </label>
            </div>
          </FormRow>

          {/* Training alerts */}
          <FormRow
            label="Email alerts"
            hint="Saved with the job, but NOT active yet - the email notifier is being rebuilt, so no alert is sent today. Use Auto-stop (above) to bound a run and watch the live loss chart on the job page."
            tip="Email delivery is currently disabled - the alert service was removed and is being replaced. These checkboxes still record your preference (so it takes effect when alerts return), but right now nothing is emailed. For now, the Auto-stop settings above are the real safeguard against an over-long run."
          >
            <div className="space-y-2">
              <Input
                type="email"
                value={alertEmail}
                onChange={(e) => {
                  // First manual edit pins the field — session won't try to
                  // refill it later. Keeps "I cleared this on purpose" working
                  // as a real opt-out.
                  setAlertEmailDirty(true)
                  setAlertEmail(e.target.value)
                }}
                placeholder={session?.user?.email ?? 'you@example.com (leave empty to opt out)'}
                className="w-full max-w-md"
              />
              <div className={`flex flex-wrap items-center gap-4 text-xs ${alertEmail.trim() ? '' : 'opacity-50 pointer-events-none'}`}>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={alertPlateau}
                    onChange={(e) => setAlertPlateau(e.target.checked)}
                    className="accent-[#7E3AF2]"
                  />
                  <span className="text-[#374151]">
                    Notify when training stalls / done
                    <span className="text-[#9ca3af]"> (best loss is 30-50+ epochs old)</span>
                  </span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={alertDiverging}
                    onChange={(e) => setAlertDiverging(e.target.checked)}
                    className="accent-[#7E3AF2]"
                  />
                  <span className="text-[#374151]">
                    Notify if loss starts going up
                    <span className="text-[#9ca3af]"> (LR or data problem)</span>
                  </span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={alertSpot}
                    onChange={(e) => setAlertSpot(e.target.checked)}
                    className="accent-[#7E3AF2]"
                  />
                  <span className="text-[#374151]">
                    Notify on spot interruption
                    <span className="text-[#9ca3af]"> (SmartCompute reclaimed the GPU)</span>
                  </span>
                </label>
              </div>
            </div>
          </FormRow>

          {/* Cost estimate */}
          <div className="mt-4 bg-gradient-to-r from-[#F7F4FC] to-[#fdf4ff] border border-[#e9d5ff] rounded-lg p-4">
            <p className="text-xs font-semibold text-[#7E3AF2] uppercase tracking-wider mb-3">Estimate</p>
            <div className="space-y-1.5 text-sm">
              <CostRow
                label="Mode"
                value={
                  mode === 'new' ? 'New (train from scratch)' :
                  mode === 'resume'   ? `Resume of ${source?.jobLabel ?? '—'}`   :
                                        `Fine-tune of ${source?.jobLabel ?? '—'}`
                }
              />
              <CostRow
                label="Compute"
                value={computeMode === 'smart'
                  ? `SmartCompute · retry budget ${maxRetriesOnInterrupt}`
                  : 'InstantCompute (warm-pool)'}
              />
              <CostRow label="GPU" value={`${selectedGpu.label} ${selectedGpu.vram} · $${hourlyRate.toFixed(2)}/hr`} />
              <CostRow label="Data" value={`${pairs} frame pairs · ${effectiveResolution}px`} />
              <CostRow label="Preset" value={`${selectedPreset.name} (${selectedPreset.model.model_type.toUpperCase()} ${effectiveModelSize})`} />
              <CostRow
                label="Steps"
                value={maxSteps > 0
                  ? `${maxSteps.toLocaleString()} max`
                  : `~${estimate.steps.toLocaleString()} (typical for ${pairs} frames)`}
              />
              <CostRow label="Estimated runtime" value={formatHourRange(estimate.lowHours, estimate.highHours)} />
              <div className="pt-2 mt-2 border-t border-[#e9d5ff] flex items-center justify-between">
                <span className="font-semibold text-[#111827]">Estimated cost</span>
                <span className="font-bold text-lg text-[#7E3AF2]">
                  ${estCostLowUSD.toFixed(0)}–${estCostHighUSD.toFixed(0)}
                </span>
              </div>
            </div>
            <details className="mt-2 text-[11px] text-[#6b7280]">
              <summary className="cursor-pointer hover:text-[#374151] select-none">
                Why this estimate?
              </summary>
              <ul className="mt-1.5 ml-4 list-disc space-y-0.5">
                {estimate.basis.map((b, i) => <li key={i}>{b}</li>)}
              </ul>
              <p className="mt-1.5 text-[10px] text-[#9ca3af]">
                Throughput numbers are unbenchmarked rough guesses by GPU class — real step/sec
                varies 2× either way with content, augmentations, and data-loader pressure.
                Spark bills per second; if your model converges early just stop the run.
              </p>
            </details>
          </div>

          {submitError && (
            <div className="mt-4 px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
              <strong>Submit failed:</strong> {submitError}
            </div>
          )}

          {hasReviewablePairs && !reviewApproved && (
            <div className="mt-4 px-4 py-3 bg-[#FFFBEB] border border-[#fde68a] rounded-lg text-sm text-[#92400E] flex items-center justify-between gap-3">
              <span>⚠ Review your image pairs before submitting.</span>
              <button
                type="button"
                onClick={() => setReviewOpen(true)}
                className="px-3 py-1.5 text-xs font-semibold rounded border border-[#7E3AF2] text-white bg-[#7E3AF2] hover:bg-[#6C2BD9] whitespace-nowrap"
              >
                Review pairs
              </button>
            </div>
          )}

          <div className="mt-5 flex items-center justify-between border-t border-[#e5e7eb] pt-5">
            <Link href="/jobs" className="text-sm text-[#6b7280] hover:text-[#374151]">
              ← Cancel
            </Link>
            <Button
              type="submit"
              loading={submitting}
              disabled={submitting || (hasReviewablePairs && !reviewApproved)}
              title={hasReviewablePairs && !reviewApproved ? 'Review and approve your image pairs first' : undefined}
              size="lg"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              {submitting ? 'Submitting…' : `Start Training · ~$${estCostLowUSD.toFixed(0)}–$${estCostHighUSD.toFixed(0)}`}
            </Button>
          </div>
        </Card>
      </form>
    </div>
    </>
  )
}

// ── Subcomponents ───────────────────────────────────────────────────────────

function Card({ step, title, info, children }: { step: number; title: string; info?: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-5 card-shadow">
      <div className="flex items-center gap-2.5 mb-4">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-[#F7F4FC] text-[#7E3AF2] text-sm font-bold">
          {step}
        </span>
        <h2 className="text-base font-semibold text-[#111827]">{title}</h2>
        {info && (
          <span className="ml-auto">
            <InfoTip text={info} align="right" />
          </span>
        )}
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

function GpuChip({ gpu, priceUsdPerHr, computeMode, selected, onClick }: {
  gpu: GpuOption
  priceUsdPerHr: number
  computeMode: ComputeMode
  selected: boolean
  onClick: () => void
}) {
  const badgeColor = (
    gpu.badge === 'recommended' ? 'text-[#ae69f4]' :
    gpu.badge === 'expensive'   ? 'text-[#D97706]' :
    gpu.badge === 'cheap'       ? 'text-[#16A34A]' :
    gpu.badge === 'fastest'     ? 'text-[#1c64f2]' :
                                  'text-[#9ca3af]'
  )
  const hourly = priceUsdPerHr
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
        <span className="text-xs text-[#374151] font-mono">
          ${hourly.toFixed(2)}/hr
          {computeMode === 'smart' && (
            <span className="ml-1 text-[9px] uppercase tracking-wider text-[#16A34A] font-bold">smart</span>
          )}
        </span>
      </div>
    </button>
  )
}

function ComputeModeToggle({ value, onChange }: {
  value: ComputeMode
  onChange: (m: ComputeMode) => void
}) {
  return (
    <div className="inline-flex rounded-lg border border-[#e5e7eb] bg-white p-0.5">
      <button
        type="button"
        onClick={() => onChange('instant')}
        className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
          value === 'instant'
            ? 'bg-[#7E3AF2] text-white'
            : 'text-[#6b7280] hover:bg-[#F9FAFB]'
        }`}
      >
        ⚡ Instant
      </button>
      <button
        type="button"
        onClick={() => onChange('smart')}
        className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
          value === 'smart'
            ? 'bg-[#16A34A] text-white'
            : 'text-[#6b7280] hover:bg-[#F9FAFB]'
        }`}
      >
        💰 Smart <span className="opacity-75">(~60% off)</span>
      </button>
    </div>
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

/**
 * Same shape as ReadonlyField but with a 4-up segmented control for one-click
 * resolution override. The preset's recommended value gets a tiny "default"
 * label; if the user has overridden, we surface that subtly via the border
 * color so it's visible-but-not-loud.
 */
const QUICK_RES = [256, 384, 512, 768] as const
function ResolutionField({
  value, presetDefault, isOverride, onChange,
}: {
  value:         number
  presetDefault: number
  isOverride:    boolean
  onChange:      (r: number) => void
}) {
  return (
    <div className={`bg-[#F9FAFB] border rounded px-3 py-2 ${
      isOverride ? 'border-[#ae69f4]' : 'border-[#e5e7eb]'
    }`}>
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">
          Resolution
        </p>
        {isOverride && (
          <span className="text-[9px] text-[#7E3AF2] uppercase tracking-wider font-semibold">
            override
          </span>
        )}
      </div>
      <div className="mt-1 flex gap-0.5 bg-white border border-[#e5e7eb] rounded p-0.5">
        {QUICK_RES.map(r => (
          <button
            key={r}
            type="button"
            onClick={() => onChange(r)}
            title={r === presetDefault ? `Preset default (${r}px)` : `Use ${r}px`}
            className={`flex-1 text-[10px] font-mono py-0.5 rounded transition-colors ${
              value === r
                ? 'bg-[#7E3AF2] text-white'
                : 'text-[#6b7280] hover:bg-[#F9FAFB]'
            }`}
          >
            {r}
            {r === presetDefault && (
              <span className={`block text-[8px] ${value === r ? 'text-white/70' : 'text-[#9ca3af]'}`}>
                default
              </span>
            )}
          </button>
        ))}
      </div>
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

function stemLower(name: string): string {
  const dot = name.lastIndexOf('.')
  return (dot < 0 ? name : name.slice(0, dot)).toLowerCase()
}

/**
 * Render a low–high hours range. Switches to minutes if both ends are < 1h
 * so we don't show "0.3h" and call it useful.
 */
function formatHourRange(low: number, high: number): string {
  if (high < 1) {
    return `${Math.round(low * 60)}–${Math.round(high * 60)} min`
  }
  if (low < 1) {
    return `${Math.round(low * 60)} min – ${high.toFixed(1)} h`
  }
  return `${low.toFixed(1)}–${high.toFixed(1)} h`
}

// uploadStage was extracted to lib/upload-stage.ts so the benchmark page can
// share the same batching logic. It's imported above.
