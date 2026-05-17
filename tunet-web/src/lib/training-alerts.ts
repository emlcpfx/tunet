/**
 * Training-alert detection — server-side.
 *
 * Pulls a job's training.log from ShareSync, parses the same Eric-format lines
 * as the live chart, runs the analyze_training heuristic from
 * training_monitor.py:649, and decides which (if any) alert kinds should fire.
 *
 * Intentionally no DB access here — the cron route owns the suppression
 * window and persistence. This module is pure: log → events.
 */

import 'server-only'
import { fetchOutputFile, type SparkJob } from './spark'

// ── Same regexes as components/spark/use-training-stream.ts ─────────────────
const TRAIN_RE = /Epoch\[(\d+)\]\s*Step\[(\d+)\](?:\((\d+)\/(\d+)\))?.*?\b(?:L1|L2|BCE\+Dice):([\d.]+)/

interface Point { x: number; y: number }

export type AlertKind = 'plateau' | 'training_done' | 'diverging'

export interface AlertSnapshot {
  currentEpoch:     number
  /** Best L1 loss observed across the whole run */
  bestLoss:         number
  bestEpoch:        number
  /** Same labels as the UI's analysis chips */
  trend:            'improving' | 'flat' | 'diverging'
  statusLabel:      string
  /** Epochs that have passed since `bestEpoch` */
  epochsSinceBest:  number
  relSlope:         number
}

export interface AnalysisResult {
  /** Null = couldn't analyze (not enough data, log missing, etc.) */
  snapshot:    AlertSnapshot | null
  /** Which alert(s) the cron should consider firing for this tick */
  recommend:   AlertKind[]
  /** Reason we couldn't analyze, populated only when snapshot is null */
  skipReason?: string
}

/**
 * Fetch + parse training.log for a job and decide which alerts qualify.
 *
 * Note: we deliberately re-fetch the whole log each tick. Spark's WebDAV
 * doesn't expose Range natively in this v1, and the file is small (~10 KB
 * per 100 epochs of training). At a 10-min cadence per active job this is
 * negligible.
 */
export async function analyzeJobForAlerts(job: SparkJob): Promise<AnalysisResult> {
  // Fetch the log
  let logText: string
  try {
    const res = await fetchOutputFile(job, 'training.log')
    if (!res.ok) {
      return { snapshot: null, recommend: [], skipReason: `log fetch HTTP ${res.status}` }
    }
    logText = await res.text()
  } catch (e) {
    return { snapshot: null, recommend: [], skipReason: e instanceof Error ? e.message : 'fetch failed' }
  }

  if (!logText) return { snapshot: null, recommend: [], skipReason: 'empty log' }

  // Parse train points only — analyze_training in training_monitor.py only
  // uses train l1, not val.
  const train: Point[] = []
  for (const line of logText.split('\n')) {
    const m = line.match(TRAIN_RE)
    if (!m) continue
    const epoch  = parseInt(m[1], 10)
    const stepIn = m[3] ? parseInt(m[3], 10) : null
    const stepTot = m[4] ? parseInt(m[4], 10) : null
    const x = stepTot && stepIn !== null ? (epoch - 1) + (stepIn / stepTot) : epoch
    const y = parseFloat(m[5])
    if (Number.isFinite(x) && Number.isFinite(y)) train.push({ x, y })
  }

  if (train.length === 0) return { snapshot: null, recommend: [], skipReason: 'no training lines parsed' }

  const currentEpoch = train[train.length - 1].x
  if (currentEpoch < 5) return { snapshot: null, recommend: [], skipReason: 'still collecting (< 5 epochs)' }

  // Best across the whole run
  let bestLoss = Infinity, bestEpoch = train[0].x
  for (const p of train) if (p.y < bestLoss) { bestLoss = p.y; bestEpoch = p.x }

  // Sliding window — same as analyze_training
  const windowEpochs = Math.min(20, currentEpoch * 0.5)
  const cutoff = currentEpoch - windowEpochs
  const window = train.filter(p => p.x >= cutoff)
  if (window.length < 10) return { snapshot: null, recommend: [], skipReason: 'window < 10 points' }

  // EMA smoothing (alpha=0.95)
  const smoothed: number[] = []
  let last = window[0].y
  for (const p of window) {
    last = 0.95 * last + 0.05 * p.y
    smoothed.push(last)
  }

  // Linear regression slope
  const n = smoothed.length
  const xs = window.map(p => p.x)
  const xMean = xs.reduce((s, v) => s + v, 0) / n
  const yMean = smoothed.reduce((s, v) => s + v, 0) / n
  let ssXy = 0, ssXx = 0
  for (let i = 0; i < n; i++) {
    ssXy += (xs[i] - xMean) * (smoothed[i] - yMean)
    ssXx += (xs[i] - xMean) ** 2
  }
  const slope = ssXx > 0 ? ssXy / ssXx : 0
  const currentSmooth = smoothed[n - 1]
  const relSlope = currentSmooth > 0 ? slope / currentSmooth : 0

  const trend: AlertSnapshot['trend'] =
    relSlope < -0.005 ? 'improving' :
    relSlope >  0.005 ? 'diverging' : 'flat'

  const epochsSinceBest = currentEpoch - bestEpoch

  // Same Status text as the UI chip — keeps alert wording aligned with what
  // the user sees on the page.
  let statusLabel: string
  if (relSlope < -0.005 && epochsSinceBest < 20) statusLabel = 'Training well'
  else if (relSlope > 0.01)                       statusLabel = 'Diverging — check LR'
  else if (epochsSinceBest > 50)                  statusLabel = 'Consider stopping'
  else if (epochsSinceBest > 30 || Math.abs(relSlope) < 0.001) statusLabel = 'Plateau — may stop'
  else if (relSlope < -0.001)                     statusLabel = 'Slow progress'
  else                                            statusLabel = 'Stable'

  const snapshot: AlertSnapshot = {
    currentEpoch, bestLoss, bestEpoch,
    trend, statusLabel, epochsSinceBest, relSlope,
  }

  // ── Decide which alerts qualify ─────────────────────────────────────────
  const recommend: AlertKind[] = []

  // Diverging — short-circuit, this is urgent
  if (relSlope > 0.01) {
    recommend.push('diverging')
    return { snapshot, recommend }
  }

  // Training done — strong "consider stopping" signal
  if (epochsSinceBest > 50) {
    recommend.push('training_done')
  }

  // Plateau — softer "may stop" signal. We only fire this if we DIDN'T fire
  // training_done; the two would be redundant in the same email.
  if (recommend.length === 0 && (epochsSinceBest > 30 || Math.abs(relSlope) < 0.001)) {
    recommend.push('plateau')
  }

  return { snapshot, recommend }
}
