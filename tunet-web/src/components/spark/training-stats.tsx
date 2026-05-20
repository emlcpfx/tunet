'use client'

/**
 * Live stats + analysis strip below the training chart. Mirrors the bottom
 * panels of training_monitor.py:
 *
 *   Analysis: | Trend | Recent Change | Plateau Check | Status |
 *   Stats:    | Loss Cur | Loss Best | Best @ Epoch | Val Loss/Best
 *             | LPIPS Cur/Best | PSNR | SSIM | Data Points | Step Time |
 *
 * The "analysis" half mirrors training_monitor.py:649 (analyze_training)
 * thresholds verbatim — keep in sync if you tweak there.
 */

import { useEffect, useMemo, useState } from 'react'
import { useTrainingStream, type Point } from './use-training-stream'
import type { SparkJob } from '@/lib/spark-types'

// Mirror of /api/spark/pricing's PricingMap shape (server-only file, can't import).
type PricingMap = Record<string, { instantUsdPerHr: number | null; smartUsdPerHr: number | null }>

interface TrainingStatsProps {
  jobId: string
  /** Used for the live cost-per-epoch projection. Optional — when absent the cost cells just render "—". */
  job?:  SparkJob
}

export function TrainingStats({ jobId, job }: TrainingStatsProps) {
  const { series, scalars, iterationsPerEpoch } = useTrainingStream(jobId)

  // ── Best tracking (across the entire run) ────────────────────────────────
  const bestL1     = useMemo(() => bestPoint(series.train), [series.train])
  const bestVal    = useMemo(() => bestPoint(series.val),   [series.val])
  const bestLpips  = useMemo(() => bestPoint(series.trainLpips), [series.trainLpips])

  // ── Analysis (last 20 epochs window — same window logic as Python) ──────
  const analysis = useMemo(() => analyzeTraining(series.train, bestL1), [series.train, bestL1])

  const cur = series.train.length > 0 ? series.train[series.train.length - 1] : null
  const lpipsCur = series.trainLpips.length > 0 ? series.trainLpips[series.trainLpips.length - 1] : null
  const valCur   = series.val.length > 0 ? series.val[series.val.length - 1] : null
  const psnrCur  = scalars.valPsnr.length > 0 ? scalars.valPsnr[scalars.valPsnr.length - 1] : null
  const ssimCur  = scalars.valSsim.length > 0 ? scalars.valSsim[scalars.valSsim.length - 1] : null
  const stepTimeAvg = avgRecent(scalars.stepTimeS, 50)

  // ── Live cost-per-epoch projection ───────────────────────────────────────
  // step_time × iterations_per_epoch × $/hr ÷ 3600. Updates as the rolling
  // step-time average converges in the early steps of a run (T/Step starts
  // pessimistic because the first batch pays warmup / cudnn-autotune cost,
  // then settles within ~30 steps). Re-projecting on every emission gives
  // the user a true running estimate rather than a stale submit-time quote.
  const [pricing, setPricing] = useState<PricingMap | null>(null)
  useEffect(() => {
    let cancelled = false
    fetch('/api/spark/pricing', { cache: 'force-cache' })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (!cancelled && d) setPricing(d as PricingMap) })
      .catch(() => { /* silent — cost cells fall back to '—' */ })
    return () => { cancelled = true }
  }, [])

  const hourlyUsd = useMemo(() => {
    if (!job || !pricing) return null
    const sku = job.instance_type_name
    if (!sku) return null
    const r = pricing[sku]
    if (!r) return null
    const rate = job.mode === 'smart' ? r.smartUsdPerHr : r.instantUsdPerHr
    return Number.isFinite(rate ?? NaN) ? rate : null
  }, [job, pricing])

  const costPerEpochUsd = (stepTimeAvg !== null && iterationsPerEpoch !== null && hourlyUsd !== null)
    ? (stepTimeAvg * iterationsPerEpoch * hourlyUsd) / 3600
    : null

  return (
    <div className="bg-white border border-[#e5e7eb] rounded-lg overflow-hidden">
      {/* Analysis row */}
      <div className="px-4 py-2 border-b border-[#e5e7eb] flex items-center gap-6 flex-wrap text-xs">
        <span className="font-semibold text-[#374151]">Analysis:</span>
        <AnalysisItem label="Trend"          value={analysis.trend.label}        color={analysis.trend.color} />
        <AnalysisItem label="Recent Change"  value={analysis.recentChange.label} color={analysis.recentChange.color} />
        <AnalysisItem label="Plateau"        value={analysis.plateau.label}      color={analysis.plateau.color} />
        <AnalysisItem label="Status"         value={analysis.status.label}       color={analysis.status.color} />
      </div>

      {/* Metric grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-px bg-[#f3f4f6]">
        <Stat label="Epoch"          value={cur ? cur.x.toFixed(2) : '—'} />
        <Stat label="Loss Current"   value={cur ? cur.y.toFixed(5) : '—'} />
        <Stat label="Loss Best"      value={bestL1 ? bestL1.y.toFixed(5) : '—'}
              hint={bestL1 ? `@ ep ${bestL1.x.toFixed(1)}` : undefined}
              accent="best" />
        <Stat label="Val Loss"       value={valCur ? valCur.y.toFixed(5) : '—'}
              hint={bestVal ? `best ${bestVal.y.toFixed(5)} @ ${bestVal.x.toFixed(0)}` : undefined} />
        <Stat label="LPIPS Cur"      value={lpipsCur ? lpipsCur.y.toFixed(5) : '—'} />
        <Stat label="LPIPS Best"     value={bestLpips ? bestLpips.y.toFixed(5) : '—'}
              hint={bestLpips ? `@ ep ${bestLpips.x.toFixed(1)}` : undefined} />
        <Stat label="PSNR (dB)"      value={psnrCur ? psnrCur.y.toFixed(2) : '—'} />
        <Stat label="SSIM"           value={ssimCur ? ssimCur.y.toFixed(4) : '—'} />
        <Stat label="Data Points"    value={series.train.length.toLocaleString()} />
        <Stat label="Step Time"      value={stepTimeAvg ? `${stepTimeAvg.toFixed(3)}s` : '—'} />
        <Stat label="Val Points"     value={series.val.length.toString()} />
        <Stat label="Best Age"       value={bestL1 && cur ? `${(cur.x - bestL1.x).toFixed(1)} ep` : '—'} />
        <Stat label="Cost / Epoch"   value={formatCost(costPerEpochUsd)}
              hint={iterationsPerEpoch ? `${iterationsPerEpoch.toLocaleString()} steps` : undefined} />
        <Stat label="Cost / 10 ep"   value={formatCost(costPerEpochUsd !== null ? costPerEpochUsd * 10 : null)}
              hint={hourlyUsd !== null ? `$${hourlyUsd.toFixed(2)}/hr` : undefined} />
      </div>
    </div>
  )
}

/**
 * Format a USD amount for the stats grid. Renders `<$0.01` for sub-penny
 * projections (rather than the misleading `$0.00`), two decimals up to $10,
 * one decimal up to $100, and whole dollars above that — keeps the cell
 * readable at all magnitudes without horizontal scroll.
 */
function formatCost(usd: number | null): string {
  if (usd === null)        return '—'
  if (usd < 0.005)         return '<$0.01'
  if (usd < 10)            return `$${usd.toFixed(2)}`
  if (usd < 100)           return `$${usd.toFixed(1)}`
  return `$${Math.round(usd)}`
}

// ─────────────────────────────────────────────────────────────────────────────
// Cells
// ─────────────────────────────────────────────────────────────────────────────

function Stat({ label, value, hint, accent }: {
  label: string; value: string; hint?: string; accent?: 'best'
}) {
  return (
    <div className="bg-white px-3 py-2 min-w-0">
      <p className="text-[9px] uppercase tracking-wider text-[#9ca3af] font-semibold truncate">{label}</p>
      <p className={`mt-0.5 text-sm font-mono font-semibold truncate ${
        accent === 'best' ? 'text-[#16A34A]' : 'text-[#111827]'
      }`}>
        {value}
      </p>
      {hint && <p className="text-[10px] text-[#9ca3af] truncate font-mono">{hint}</p>}
    </div>
  )
}

function AnalysisItem({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-[#9ca3af]">{label}</span>
      <span className={`font-mono font-semibold ${color}`}>{value}</span>
    </span>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure helpers
// ─────────────────────────────────────────────────────────────────────────────

function bestPoint(pts: Point[]): Point | null {
  if (pts.length === 0) return null
  let best = pts[0]
  for (const p of pts) if (p.y < best.y) best = p
  return best
}

function avgRecent(pts: Point[], n: number): number | null {
  if (pts.length === 0) return null
  const slice = pts.slice(-n)
  return slice.reduce((s, p) => s + p.y, 0) / slice.length
}

interface AnalysisLabel { label: string; color: string }

interface Analysis {
  trend:         AnalysisLabel
  recentChange:  AnalysisLabel
  plateau:       AnalysisLabel
  status:        AnalysisLabel
}

const COLORS = {
  good: 'text-[#16A34A]',
  warn: 'text-[#D97706]',
  bad:  'text-[#EF4444]',
  dim:  'text-[#9ca3af]',
  text: 'text-[#374151]',
}

/**
 * Mirror of training_monitor.py:649 analyze_training. Same windowing, same
 * EMA, same slope-based heuristics + thresholds. Returns labelled values
 * for the four analysis chips.
 */
function analyzeTraining(train: Point[], bestL1: Point | null): Analysis {
  const collecting: Analysis = {
    trend:        { label: 'Collecting…', color: COLORS.dim },
    recentChange: { label: 'Collecting…', color: COLORS.dim },
    plateau:      { label: 'Collecting…', color: COLORS.dim },
    status:       { label: 'Collecting…', color: COLORS.dim },
  }
  if (train.length === 0) return collecting

  const currentEpoch = train[train.length - 1].x
  if (currentEpoch < 5) return collecting

  // Sliding window: last min(20, half-of-run) epochs
  const windowEpochs = Math.min(20, currentEpoch * 0.5)
  const cutoff = currentEpoch - windowEpochs
  const windowPts = train.filter(p => p.x >= cutoff)
  if (windowPts.length < 10) return collecting

  // EMA (alpha = 0.95)
  const smoothed: number[] = []
  let last = windowPts[0].y
  for (const p of windowPts) {
    last = 0.95 * last + 0.05 * p.y
    smoothed.push(last)
  }

  // Linear regression slope on (epoch, smoothed)
  const n = smoothed.length
  const xs = windowPts.map(p => p.x)
  const ys = smoothed
  const xMean = xs.reduce((s, v) => s + v, 0) / n
  const yMean = ys.reduce((s, v) => s + v, 0) / n
  let ssXy = 0, ssXx = 0
  for (let i = 0; i < n; i++) {
    ssXy += (xs[i] - xMean) * (ys[i] - yMean)
    ssXx += (xs[i] - xMean) ** 2
  }
  const slope = ssXx > 0 ? ssXy / ssXx : 0
  const currentSmooth = smoothed[n - 1]
  const relSlope = currentSmooth > 0 ? slope / currentSmooth : 0

  // ── Recent Change (% diff first vs second half of window) ────────────────
  // Half-vs-half mean is more robust to EMA end-lag than a regression slope:
  // when raw loss drops fast inside the window, the smoothed series starts
  // near raw[0] and only gradually catches up, so `slope` reads near-zero
  // even though the cumulative drop is large. Half-vs-half captures that
  // cumulative drop honestly. We use it as the canonical "is loss going
  // down?" signal — both the Trend chip and the Status recommendation read
  // from it (so they can never disagree like they did when Trend was
  // computed straight off `relSlope`).
  const mid = Math.floor(smoothed.length / 2)
  const firstAvg  = mid > 0 ? smoothed.slice(0, mid).reduce((s, v) => s + v, 0) / mid : 0
  const secondAvg = (smoothed.length - mid) > 0
    ? smoothed.slice(mid).reduce((s, v) => s + v, 0) / (smoothed.length - mid) : 0
  const pct = firstAvg > 0 ? ((secondAvg - firstAvg) / firstAvg) * 100 : 0
  const pctColor = pct < -1 ? COLORS.good : pct > 1 ? COLORS.bad : COLORS.warn
  const recentChange: AnalysisLabel = {
    label: `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`,
    color: pctColor,
  }

  // ── Trend ────────────────────────────────────────────────────────────────
  // Derived from the same `pct` metric as Recent Change so the two chips
  // can never contradict (e.g. "Trend: Flat" alongside "Recent Change: -29.9%"
  // used to happen because Trend read a near-zero regression slope while
  // Recent Change read the lag-aware mean comparison).
  const trend: AnalysisLabel =
    pct < -1 ? { label: 'Improving', color: COLORS.good } :
    pct >  1 ? { label: 'Diverging', color: COLORS.bad } :
               { label: 'Flat',      color: COLORS.warn }

  // ── Plateau ──────────────────────────────────────────────────────────────
  const epochsSinceBest = bestL1 ? currentEpoch - bestL1.x : 0
  const plateauColor =
    epochsSinceBest < 10 ? COLORS.good :
    epochsSinceBest < 30 ? COLORS.warn :
                           COLORS.bad
  const plateau: AnalysisLabel = {
    label: `Best ${epochsSinceBest.toFixed(0)}ep ago`,
    color: plateauColor,
  }

  // ── Status (the only chip a beginner needs to act on) ───────────────────
  // Decision tree in priority order. Earlier branches override later ones.
  //
  //   1. Diverging (>+1%/ep slope)  → "stop, lower LR" regardless of best-age
  //   2. New best just landed       → never "stop", even if slope reads flat
  //                                    (slope lags by design when loss drops fast)
  //   3. Active improvement         → "training well"
  //   4. Stale best (>50ep)         → strong stop signal
  //   5. Stale-ish (>30ep)          → soft stop signal
  //   6. Flat with old-ish best     → "slowing, keep watching"
  //   7. Mild improvement           → "slow improvement"
  //   8. None of the above          → "stable"
  //
  // Guard #2 is the load-bearing one: without it, a fresh best with tiny
  // smoothed slope ("Flat" trend) would mis-recommend stopping.
  const STRONG_IMPROVE = pct < -3       // half-over-half drop > 3 %
  const DIVERGING      = relSlope > 0.01 || pct > 3

  let status: AnalysisLabel
  if (DIVERGING) {
    status = { label: 'Stop — diverging, lower LR', color: COLORS.bad }
  } else if (epochsSinceBest <= 2) {
    status = { label: 'Training well — new best', color: COLORS.good }
  } else if (relSlope < -0.005 || STRONG_IMPROVE) {
    status = { label: 'Training well', color: COLORS.good }
  } else if (epochsSinceBest > 50) {
    status = { label: 'Stop — no new best in 50+ epochs', color: COLORS.bad }
  } else if (epochsSinceBest > 30) {
    status = { label: 'Plateau — may stop', color: COLORS.warn }
  } else if (Math.abs(relSlope) < 0.001 && epochsSinceBest > 10) {
    status = { label: 'Slowing — keep watching', color: COLORS.warn }
  } else if (relSlope < -0.001) {
    status = { label: 'Slow improvement', color: COLORS.text }
  } else {
    status = { label: 'Stable', color: COLORS.text }
  }

  return { trend, recentChange, plateau, status }
}
