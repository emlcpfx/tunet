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

import { useMemo } from 'react'
import { useTrainingStream, type Point } from './use-training-stream'

interface TrainingStatsProps {
  jobId: string
}

export function TrainingStats({ jobId }: TrainingStatsProps) {
  const { series, scalars } = useTrainingStream(jobId)

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
      </div>
    </div>
  )
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

  // ── Trend ────────────────────────────────────────────────────────────────
  const trend: AnalysisLabel =
    relSlope < -0.005 ? { label: 'Improving', color: COLORS.good } :
    relSlope >  0.005 ? { label: 'Diverging', color: COLORS.bad } :
                        { label: 'Flat',      color: COLORS.warn }

  // ── Recent Change (% diff first vs second half of window) ────────────────
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

  // ── Status (recommendation) ─────────────────────────────────────────────
  let status: AnalysisLabel
  if (relSlope < -0.005 && epochsSinceBest < 20) {
    status = { label: 'Training well', color: COLORS.good }
  } else if (relSlope > 0.01) {
    status = { label: 'Diverging — check LR', color: COLORS.bad }
  } else if (epochsSinceBest > 50) {
    status = { label: 'Consider stopping', color: COLORS.bad }
  } else if (epochsSinceBest > 30 || Math.abs(relSlope) < 0.001) {
    status = { label: 'Plateau — may stop', color: COLORS.warn }
  } else if (relSlope < -0.001) {
    status = { label: 'Slow progress', color: COLORS.warn }
  } else {
    status = { label: 'Stable', color: COLORS.text }
  }

  return { trend, recentChange, plateau, status }
}
