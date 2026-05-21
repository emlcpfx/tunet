'use client'

/**
 * Shared SSE-stream parser for tunet training logs. Multiple components on the
 * same job page can subscribe to one stream — the hook dedupes by jobId and
 * holds a single EventSource per process, distributing parsed series + scalar
 * metrics to all subscribers.
 *
 * Why a singleton: EventSource is cheap (the proxy replays from start each
 * time), but parallel SSE connections multiply Spark API load AND every
 * subscriber re-runs the regex on every line. Sharing keeps it linear.
 *
 * Parses tunet's training format (matches train.py:1617):
 *   "Epoch[3] Step[42] (125/500), L1:0.0142 ..., LPIPS:0.0089 ..., T/Step:0.069s"
 *   "Val Epoch[3] Step[42] Val_L1:0.0156 Val_LPIPS:0.0091 PSNR:31.2dB SSIM:0.94"
 * Note the space + comma around the (stepIn/stepTot) group — the regex below
 * tolerates optional whitespace there.
 */

import { useEffect, useState } from 'react'

export interface Point  { x: number; y: number }
export interface Series { train: Point[]; val: Point[]; trainLpips: Point[]; valLpips: Point[] }

export interface ScalarSeries {
  /** Average per-step wall-clock time across the live window */
  stepTimeS:   Point[]      // x = epoch, y = seconds
  valPsnr:     Point[]      // x = epoch, y = dB
  valSsim:     Point[]      // x = epoch, y = [0,1]
}

export interface TrainingStreamState {
  series:    Series
  scalars:   ScalarSeries
  status:    'connecting' | 'streaming' | 'closed' | 'error'
  /**
   * Most recently observed `iterations_per_epoch` from a train log line's
   * `(stepIn/stepTot)` suffix. Null until the first such line lands. Used
   * by callers that need to convert per-step quantities (e.g. step time,
   * per-step cost) into per-epoch quantities.
   */
  iterationsPerEpoch: number | null
  /**
   * Latest cumulative `global_step` from a train line's `Step[N]` (train.py
   * keeps this counting across a resume). Null until the first train line.
   */
  currentStep: number | null
  /**
   * Latest cumulative epoch number from a train line's `Epoch[N]` (1-indexed,
   * in-progress; also continues across a resume). Null until first train line.
   */
  currentEpoch: number | null
  /**
   * Global-step baseline this run resumed from, parsed from train.py's
   * "Resuming from Global Step N (Logical Epoch M)" line (train.py:872).
   * Null for fresh / fine-tune runs (which start global_step at 0). When set,
   * `currentStep - resumeFromStep` is "steps since resume".
   */
  resumeFromStep:  number | null
  /** Logical-epoch baseline (1-indexed) from the same resume line; null if not a resume. */
  resumeFromEpoch: number | null
}

const EMPTY_STREAM: TrainingStreamState = {
  series:  { train: [], val: [], trainLpips: [], valLpips: [] },
  scalars: { stepTimeS: [], valPsnr: [], valSsim: [] },
  status:  'connecting',
  iterationsPerEpoch: null,
  currentStep:     null,
  currentEpoch:    null,
  resumeFromStep:  null,
  resumeFromEpoch: null,
}

const TRAIN_RE = /Epoch\[(\d+)\]\s*Step\[(\d+)\]\s*(?:\((\d+)\/(\d+)\))?.*?\b(L1|L2|BCE\+Dice):([\d.]+)/
const LPIPS_RE = /LPIPS:([\d.]+)/
const TIME_RE  = /T\/Step:([\d.]+)s/
const VAL_RE   = /Val Epoch\[(\d+)\].*?Val_(L1|L2|BCE\+Dice):([\d.]+)/
const VAL_LPIPS_RE = /Val_LPIPS:([\d.]+)/
const VAL_PSNR_RE  = /PSNR:([\d.]+)dB/
const VAL_SSIM_RE  = /SSIM:([\d.]+)/
const RESUME_RE    = /Resuming from Global Step (\d+) \(Logical Epoch (\d+)\)/

interface LogLine {
  ts:     string
  stream: 'stdout' | 'stderr'
  phase:  'agent' | 'container'
  line:   string
}

// ── Singleton store keyed by jobId ──────────────────────────────────────────
//
// We can't use React context for this because the hook runs on multiple
// components in parallel and we want O(1) lookup. Plain module-level Map +
// listener set is the smallest thing that works.
interface Store {
  state:        TrainingStreamState
  subscribers:  Set<(s: TrainingStreamState) => void>
  es:           EventSource | null
  refCount:     number
}
const stores = new Map<string, Store>()

function getStore(jobId: string): Store {
  let s = stores.get(jobId)
  if (!s) {
    s = { state: EMPTY_STREAM, subscribers: new Set(), es: null, refCount: 0 }
    stores.set(jobId, s)
  }
  return s
}

function emit(store: Store, next: TrainingStreamState) {
  store.state = next
  for (const fn of store.subscribers) fn(next)
}

function openStream(jobId: string) {
  const store = getStore(jobId)
  if (store.es) return // already open

  const es = new EventSource(`/api/spark/jobs/${jobId}/logs`)
  store.es = es

  es.onopen = () => emit(store, { ...store.state, status: 'streaming' })

  const onLog = (ev: MessageEvent) => {
    let parsed: LogLine
    try { parsed = JSON.parse(ev.data) as LogLine } catch { return }
    if (parsed.phase !== 'container') return
    const line = parsed.line
    const cur  = store.state

    // Resume baseline (one-shot, early in the run). Cheap regex, rare line —
    // checked before the train branch so it can't be shadowed by it.
    const rm = line.match(RESUME_RE)
    if (rm) {
      emit(store, {
        ...cur,
        resumeFromStep:  parseInt(rm[1], 10),
        resumeFromEpoch: parseInt(rm[2], 10),
      })
      return
    }

    const tm = line.match(TRAIN_RE)
    if (tm) {
      const epoch    = parseInt(tm[1], 10)
      const stepIn   = tm[3] ? parseInt(tm[3], 10) : null
      const stepTot  = tm[4] ? parseInt(tm[4], 10) : null
      const x        = stepTot && stepIn !== null
        ? (epoch - 1) + (stepIn / stepTot)
        : epoch
      const y        = parseFloat(tm[6])
      const series   = { ...cur.series, train: [...cur.series.train, { x, y }] }
      const lp       = line.match(LPIPS_RE)
      if (lp) series.trainLpips = [...cur.series.trainLpips, { x, y: parseFloat(lp[1]) }]

      let scalars = cur.scalars
      const tt = line.match(TIME_RE)
      if (tt) scalars = { ...scalars, stepTimeS: [...scalars.stepTimeS, { x, y: parseFloat(tt[1]) }] }

      // Track iterations-per-epoch from the (stepIn/stepTot) suffix. Spark
      // training emits this on every line; we just keep the latest value
      // because progressive_resolution can change it mid-run.
      const iterationsPerEpoch = stepTot && stepTot > 0 ? stepTot : cur.iterationsPerEpoch

      // Step[N]/Epoch[N] are cumulative (continue across a resume).
      const currentStep  = parseInt(tm[2], 10)
      const currentEpoch = epoch

      emit(store, { ...cur, series, scalars, iterationsPerEpoch, currentStep, currentEpoch })
      return
    }

    const vm = line.match(VAL_RE)
    if (vm) {
      const epoch  = parseInt(vm[1], 10)
      const y      = parseFloat(vm[3])
      const series = { ...cur.series, val: [...cur.series.val, { x: epoch, y }] }
      const vlp = line.match(VAL_LPIPS_RE)
      if (vlp) series.valLpips = [...cur.series.valLpips, { x: epoch, y: parseFloat(vlp[1]) }]

      let scalars = cur.scalars
      const ps = line.match(VAL_PSNR_RE)
      if (ps) scalars = { ...scalars, valPsnr: [...scalars.valPsnr, { x: epoch, y: parseFloat(ps[1]) }] }
      const ss = line.match(VAL_SSIM_RE)
      if (ss) scalars = { ...scalars, valSsim: [...scalars.valSsim, { x: epoch, y: parseFloat(ss[1]) }] }

      emit(store, { ...cur, series, scalars })
    }
  }

  es.addEventListener('log', onLog as EventListener)
  es.addEventListener('message', onLog as EventListener)
  es.addEventListener('error', () => {
    emit(store, { ...store.state, status: es.readyState === EventSource.CLOSED ? 'closed' : 'error' })
  })
}

function closeStream(jobId: string) {
  const store = stores.get(jobId)
  if (!store) return
  store.es?.close()
  store.es = null
  stores.delete(jobId)
}

/**
 * Subscribe a component to the shared parsed stream for a job.
 *
 * The first subscriber opens the EventSource; the last unsubscribe closes
 * it. New subscribers immediately receive the current accumulated state.
 */
export function useTrainingStream(jobId: string): TrainingStreamState {
  const [state, setState] = useState<TrainingStreamState>(() => getStore(jobId).state)

  useEffect(() => {
    const store = getStore(jobId)
    store.refCount += 1
    store.subscribers.add(setState)
    setState(store.state)
    if (!store.es) openStream(jobId)

    return () => {
      const s = stores.get(jobId)
      if (!s) return
      s.subscribers.delete(setState)
      s.refCount -= 1
      if (s.refCount <= 0) closeStream(jobId)
    }
  }, [jobId])

  return state
}
