'use client'

/**
 * /benchmark — calibration page for the cost estimator.
 *
 * Submits one short job per GPU at the reference settings (UNet, model_size
 * 64, 512px, batch 2, L1 loss). Each job runs ~200 steps after warmup, logs
 * STEP_RATE: X.Y, and exits — typically <2 minutes wall, <$0.20 each.
 *
 * For each in-flight run we open the SSE log stream and grep for STEP_RATE.
 * Once all five GPUs have reported, the "Copy as TS code" button at the
 * bottom emits a ready-to-paste body for `baselineStepsPerSec()` in
 * spark-presets.ts.
 *
 * Deliberately separate from the new-job form: the benchmark is a one-off
 * dev/admin operation, not part of the user training flow.
 */

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { pricePerHour } from '@/lib/spark-presets'
import { FolderPicker, type FolderPickerResult } from '@/components/spark/folder-picker'
import { uploadStage } from '@/lib/upload-stage'

interface GpuOption { key: string; sku: string; label: string; vram: string }

// Same order as the new-job page (cheapest → fastest, A10 = recommended).
const BENCH_GPUS: GpuOption[] = [
  { key: 't4',         sku: 'g4dn.xlarge', label: 'T4',           vram: '16GB' },
  { key: 'l4',         sku: 'g6.2xlarge',  label: 'L4',           vram: '24GB' },
  { key: 'a10',        sku: 'g5.xlarge',   label: 'A10',          vram: '24GB' },
  { key: 'rtxpro6000', sku: 'g7e.2xlarge', label: 'RTX PRO 6000', vram: '96GB' },
  { key: 'l40s',       sku: 'g6e.8xlarge', label: 'L40S',         vram: '48GB' },
]

interface RunState {
  gpuKey:        string
  sku:           string
  label:         string
  jobId:         string
  resolution:    number
  batchSize:     number
  startedAt:     number
  // Streaming state
  status:        'queued' | 'streaming' | 'measured' | 'failed' | 'closed'
  rate:          number | null    // step/sec from the parsed STEP_RATE line
  rateLineRaw:   string | null    // full log line for the audit trail
  errorMsg:      string | null
}

const STEP_RATE_RE = /STEP_RATE:\s*([\d.]+)\s*step\/sec/

// Sweep axes. Defaults match the server route. Skip batch=1 (always
// sublinear → not informative for picking a *good* batch) and batch=16
// (almost certainly OOM on T4 16GB at high res). User can adjust via the
// checkbox grids on the page.
const RESOLUTIONS = [256, 512, 1024]
const BATCH_SIZES = [2, 4, 8]

export default function BenchmarkPage() {
  const [selected, setSelected] = useState<Set<string>>(() => new Set(BENCH_GPUS.map(g => g.key)))
  const [resSelected,   setResSelected]   = useState<Set<number>>(() => new Set(RESOLUTIONS))
  const [batchSelected, setBatchSelected] = useState<Set<number>>(() => new Set(BATCH_SIZES))
  const [steps,   setSteps]     = useState(200)
  const [warmup,  setWarmup]    = useState(20)
  const [runs,    setRuns]      = useState<RunState[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Optional real-data calibration: user picks a folder of EXR/PNG pairs,
  // we upload-stage it, and pass the stageId through to the benchmark route
  // so the dataloader I/O matches a real run. Synthetic remains the default
  // for "click and walk away" calibration.
  const [picked, setPicked]           = useState<FolderPickerResult | null>(null)
  const [stageId, setStageId]         = useState<string | null>(null)
  const [staging, setStaging]         = useState<{ sent: number; total: number } | null>(null)
  const [stageError, setStageError]   = useState<string | null>(null)
  // When the user re-picks, invalidate any prior stageId so the next run
  // re-uploads (different File objects, possibly different content).
  useEffect(() => { setStageId(null) }, [picked])

  // Spark's allow-list of compute SKUs varies by account/region. Fetch it on
  // mount so we can disable any GPU rows that submit would refuse anyway.
  const [allowedSkus, setAllowedSkus]   = useState<Set<string> | null>(null)
  const [skuError, setSkuError]         = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/spark/skus', { cache: 'no-store' })
        const data = await res.json()
        if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
        if (cancelled) return
        const set = new Set<string>((data.skus ?? []).map((s: { instanceType: string }) => s.instanceType))
        setAllowedSkus(set)
        // Auto-deselect any GPU that isn't currently eligible so the cost
        // estimate and submit button only count what'll actually run.
        setSelected(prev => {
          const next = new Set<string>()
          for (const k of prev) {
            const g = BENCH_GPUS.find(x => x.key === k)
            if (g && set.has(g.sku)) next.add(k)
          }
          return next
        })
      } catch (e) {
        if (!cancelled) setSkuError(e instanceof Error ? e.message : 'failed to load SKUs')
      }
    })()
    return () => { cancelled = true }
  }, [])

  // Toggle a GPU
  const toggle = (k: string) => setSelected(prev => {
    const next = new Set(prev)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    return next
  })

  // Cells we'll actually submit (cartesian product). Useful for the cost
  // line, the cap warning, and the empty-state in the results table.
  const cellCount = selected.size * resSelected.size * batchSelected.size

  // Estimated total cost: ~90 sec per cell at 512px (warmup + 200 steps),
  // scaled by (res/512)² for higher res, with a +50% safety margin since
  // some cells will OOM and exit fast (cheaper) but provisioning + cold
  // pull dominate when stuff queues serially.
  const totalCost = useMemo(() => {
    let usd = 0
    for (const k of selected) {
      const g = BENCH_GPUS.find(x => x.key === k)
      if (!g) continue
      const hourly = pricePerHour(g.sku)
      for (const r of resSelected) {
        const resPenalty = (r / 512) ** 2
        const cellSeconds = 90 * resPenalty * 1.5
        usd += (cellSeconds / 3600) * hourly * batchSelected.size
      }
    }
    return usd
  }, [selected, resSelected, batchSelected])

  async function runBenchmark() {
    if (selected.size === 0 || submitting) return
    setSubmitting(true)
    setSubmitError(null)
    setStageError(null)
    setRuns([])
    try {
      // Stage the user's data first if they picked a folder and we haven't
      // already uploaded it. Re-uses the staged dir across all N benchmark
      // jobs (same dataset → apples-to-apples GPU comparison).
      let useStageId = stageId
      if (picked && picked.src.length > 0 && picked.dst.length > 0 && !useStageId) {
        try {
          const totalBytes = picked.src.reduce((s, f) => s + f.size, 0)
                          + picked.dst.reduce((s, f) => s + f.size, 0)
          setStaging({ sent: 0, total: totalBytes })
          // Strip val/mask — benchmark only needs src+dst.
          const minimal: FolderPickerResult = { ...picked, valSrc: [], valDst: [], mask: [] }
          useStageId = await uploadStage(minimal, (sent) => {
            setStaging({ sent, total: totalBytes })
          })
          setStageId(useStageId)
        } catch (e) {
          setStageError(e instanceof Error ? e.message : 'stage upload failed')
          return
        } finally {
          setStaging(null)
        }
      }

      const res = await fetch('/api/spark/benchmark', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          gpus:        BENCH_GPUS.map(g => g.key).filter(k => selected.has(k)),
          // Send sorted so the results matrix has stable column order.
          resolutions: [...resSelected].sort((a, b) => a - b),
          batchSizes:  [...batchSelected].sort((a, b) => a - b),
          benchmarkSteps:  steps,
          benchmarkWarmup: warmup,
          ...(useStageId ? { stageId: useStageId } : {}),
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
      const newRuns: RunState[] = (data.runs ?? []).map((r: { gpuKey: string; sku: string; jobId: string; label: string; resolution: number; batchSize: number }) => ({
        gpuKey:      r.gpuKey,
        sku:         r.sku,
        label:       r.label,
        jobId:       r.jobId,
        resolution:  r.resolution,
        batchSize:   r.batchSize,
        startedAt:   Date.now(),
        status:      'streaming',
        rate:        null,
        rateLineRaw: null,
        errorMsg:    null,
      }))
      setRuns(newRuns)
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : 'submit failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-5 animate-slide-in max-w-4xl">
      <div>
        <Link href="/" className="text-sm text-[#6b7280] hover:text-[#374151]">← Demo</Link>
        <h1 className="text-2xl font-bold text-[#111827] mt-2">GPU Throughput Benchmark</h1>
        <p className="text-sm text-[#6b7280] mt-1">
          Calibrates the cost estimator. Runs ~200 timed training steps per GPU
          at fixed reference settings (UNet, 64-dim, 512px, batch 2, L1 loss),
          logs <code className="font-mono bg-[#F9FAFB] px-1 rounded">STEP_RATE</code>,
          and exits. Total wall-clock ~2 min per GPU. Pick a real EXR folder
          below for an honest baseline; otherwise we use synthetic PNGs.
        </p>
      </div>

      <Card>
        <h2 className="text-sm font-semibold text-[#111827] mb-2">Dataset</h2>
        <p className="text-xs text-[#6b7280] mb-3">
          Default: 8 synthetic 768×768 PNG pairs generated server-side.
          Fast and zero-config but PNG decode is much faster than EXR — the
          calibrated baseline ends up optimistic. <strong>Pick a folder of
          your own EXR/PNG pairs</strong> for realistic dataloader I/O.
        </p>
        <FolderPicker onPicked={setPicked} />
        {staging && (
          <div className="mt-3">
            <div className="h-1.5 bg-[#e5e7eb] rounded overflow-hidden">
              <div
                className="h-full bg-[#7E3AF2] transition-all"
                style={{ width: `${staging.total ? (staging.sent / staging.total) * 100 : 0}%` }}
              />
            </div>
            <p className="text-[11px] text-[#7E3AF2] font-mono mt-1">
              Uploading dataset… {formatBytes(staging.sent)} / {formatBytes(staging.total)}
            </p>
          </div>
        )}
        {stageError && (
          <p className="mt-2 text-xs text-[#EF4444]">Upload failed: {stageError}</p>
        )}
        {picked && stageId && !staging && (
          <p className="mt-2 text-[11px] text-[#16A34A]">
            ✓ Staged. All benchmark jobs will share this dataset.
          </p>
        )}
      </Card>

      <Card>
        <h2 className="text-sm font-semibold text-[#111827] mb-3">GPUs to benchmark</h2>
        {skuError && (
          <p className="text-xs text-[#D97706] mb-2">
            Couldn&apos;t load eligible SKUs: {skuError}. All GPUs shown — some may fail at submit.
          </p>
        )}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {BENCH_GPUS.map(g => {
            const on = selected.has(g.key)
            // null = still loading; ineligible only when we have a list and
            // the SKU isn't in it. While loading, render as enabled so the
            // user isn't blocked staring at a disabled grid.
            const eligible = allowedSkus == null ? true : allowedSkus.has(g.sku)
            return (
              <label
                key={g.key}
                className={`flex items-center gap-2 px-3 py-2 rounded border-2 transition-all ${
                  !eligible ? 'border-[#e5e7eb] bg-[#F9FAFB] opacity-60 cursor-not-allowed' :
                  on        ? 'border-[#7E3AF2] bg-[#F7F4FC] cursor-pointer' :
                              'border-[#e5e7eb] bg-white hover:border-[#D1D5DB] cursor-pointer'
                }`}
                title={!eligible ? 'Not eligible on this Spark account/region' : undefined}
              >
                <input
                  type="checkbox"
                  checked={on && eligible}
                  disabled={!eligible}
                  onChange={() => eligible && toggle(g.key)}
                  className="accent-[#7E3AF2]"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-[#111827]">
                    {g.label}
                    {!eligible && <span className="ml-1 text-[10px] uppercase text-[#D97706]">unavailable</span>}
                  </p>
                  <p className="text-[11px] text-[#6b7280]">{g.vram} · ${pricePerHour(g.sku).toFixed(2)}/hr</p>
                </div>
              </label>
            )
          })}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 text-sm border-t border-[#e5e7eb] pt-4">
          <div>
            <p className="text-xs font-semibold text-[#374151] uppercase tracking-wider mb-2">Resolutions</p>
            <div className="flex gap-2 flex-wrap">
              {RESOLUTIONS.map(r => {
                const on = resSelected.has(r)
                return (
                  <button
                    key={r}
                    type="button"
                    onClick={() => setResSelected(prev => {
                      const next = new Set(prev)
                      if (next.has(r)) next.delete(r); else next.add(r)
                      return next
                    })}
                    className={`px-3 py-1.5 rounded text-sm font-mono border-2 transition-all ${
                      on ? 'border-[#7E3AF2] bg-[#F7F4FC] text-[#7E3AF2]'
                         : 'border-[#e5e7eb] bg-white text-[#6b7280] hover:border-[#D1D5DB]'
                    }`}
                  >
                    {r}px
                  </button>
                )
              })}
            </div>
          </div>
          <div>
            <p className="text-xs font-semibold text-[#374151] uppercase tracking-wider mb-2">Batch sizes</p>
            <div className="flex gap-2 flex-wrap">
              {BATCH_SIZES.map(b => {
                const on = batchSelected.has(b)
                return (
                  <button
                    key={b}
                    type="button"
                    onClick={() => setBatchSelected(prev => {
                      const next = new Set(prev)
                      if (next.has(b)) next.delete(b); else next.add(b)
                      return next
                    })}
                    className={`px-3 py-1.5 rounded text-sm font-mono border-2 transition-all ${
                      on ? 'border-[#7E3AF2] bg-[#F7F4FC] text-[#7E3AF2]'
                         : 'border-[#e5e7eb] bg-white text-[#6b7280] hover:border-[#D1D5DB]'
                    }`}
                  >
                    bs={b}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-[#374151]">Timed steps:</span>
            <input
              type="number" min={50} max={2000} step={50}
              value={steps}
              onChange={e => setSteps(parseInt(e.target.value, 10) || 200)}
              className="w-20 border border-[#e5e7eb] rounded px-2 py-1 font-mono"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-[#374151]">Warmup:</span>
            <input
              type="number" min={5} max={200} step={5}
              value={warmup}
              onChange={e => setWarmup(parseInt(e.target.value, 10) || 20)}
              className="w-16 border border-[#e5e7eb] rounded px-2 py-1 font-mono"
            />
          </label>
          <span className="text-[11px] text-[#9ca3af] ml-auto">
            <span className="font-mono text-[#374151]">{cellCount}</span> cells ·
            est. <span className="font-mono text-[#374151]">~${totalCost.toFixed(2)}</span>
          </span>
        </div>

        <div className="mt-4 flex items-center gap-3 border-t border-[#e5e7eb] pt-3">
          <button
            type="button"
            onClick={runBenchmark}
            disabled={submitting || cellCount === 0 || cellCount > 60}
            className="px-4 py-2 text-sm font-semibold bg-[#7E3AF2] hover:bg-[#6C2BD9] text-white rounded disabled:opacity-50"
          >
            {submitting
              ? 'Submitting…'
              : cellCount > 60
                ? `Too many cells (${cellCount}, cap 60)`
                : `Run benchmark · ${cellCount} job${cellCount === 1 ? '' : 's'}`}
          </button>
          {submitError && <span className="text-xs text-[#EF4444]">Error: {submitError}</span>}
        </div>
      </Card>

      {runs.length > 0 && (
        <Card>
          <h2 className="text-sm font-semibold text-[#111827] mb-3">Results</h2>
          <ResultsTable runs={runs} setRuns={setRuns} />
        </Card>
      )}

      {runs.length > 0 && runs.every(r => r.status === 'measured' || r.status === 'failed') && (
        <Card>
          <h2 className="text-sm font-semibold text-[#111827] mb-2">Calibrated baseline</h2>
          <p className="text-xs text-[#6b7280] mb-3">
            Paste the body below into <code className="font-mono bg-[#F9FAFB] px-1 rounded">baselineStepsPerSec()</code> in{' '}
            <code className="font-mono bg-[#F9FAFB] px-1 rounded">tunet-web/src/lib/spark-presets.ts</code>.
            Failed measurements fall back to the previous guess so the function
            still has a value for unmeasured SKUs.
          </p>
          <CodeBlock code={emitTsBaseline(runs)} />
        </Card>
      )}
    </div>
  )
}

// ── Subcomponents ────────────────────────────────────────────────────────────

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-white border border-[#e5e7eb] rounded-xl p-5 card-shadow">
      {children}
    </div>
  )
}

function ResultsTable({ runs, setRuns }: { runs: RunState[]; setRuns: (f: (r: RunState[]) => RunState[]) => void }) {
  // We still need to attach the per-run polling/streaming effect to every
  // run, but don't render them as a flat list — render as a per-GPU matrix
  // below. This invisible block handles the side-effects.
  const sideEffects = runs.map(r => <RunSideEffects key={r.jobId} run={r} setRuns={setRuns} />)

  // Group by GPU, then within each GPU build a res × batch grid.
  const byGpu = new Map<string, RunState[]>()
  for (const r of runs) {
    if (!byGpu.has(r.gpuKey)) byGpu.set(r.gpuKey, [])
    byGpu.get(r.gpuKey)!.push(r)
  }

  // Stable axes — distinct values across all runs, sorted.
  const allRes   = [...new Set(runs.map(r => r.resolution))].sort((a, b) => a - b)
  const allBatch = [...new Set(runs.map(r => r.batchSize))].sort((a, b) => a - b)

  return (
    <div>
      <div style={{ display: 'none' }}>{sideEffects}</div>
      <div className="space-y-5">
        {[...byGpu.entries()].map(([gpuKey, gpuRuns]) => (
          <GpuMatrix
            key={gpuKey}
            gpuKey={gpuKey}
            label={gpuRuns[0].label}
            sku={gpuRuns[0].sku}
            runs={gpuRuns}
            allRes={allRes}
            allBatch={allBatch}
          />
        ))}
      </div>
    </div>
  )
}

/**
 * One GPU's slice of the 3-axis matrix: rows = resolutions, cols = batches.
 * Each cell shows step/sec (raw) and samples/sec (= step/sec × batch),
 * status pill, and an OOM/failed marker. Clicking a cell jumps to that
 * job's detail page.
 */
function GpuMatrix({
  gpuKey, label, sku, runs, allRes, allBatch,
}: {
  gpuKey: string
  label:  string
  sku:    string
  runs:   RunState[]
  allRes: number[]
  allBatch: number[]
}) {
  // Index runs by (res, batch) for O(1) lookup
  const cell = new Map<string, RunState>()
  for (const r of runs) cell.set(`${r.resolution}|${r.batchSize}`, r)
  void gpuKey   // referenced only via key prop in parent

  return (
    <div className="border border-[#e5e7eb] rounded-lg overflow-hidden">
      <div className="px-4 py-2 bg-[#fafafa] border-b border-[#e5e7eb]">
        <p className="text-sm font-semibold text-[#111827]">{label}</p>
        <p className="text-[11px] text-[#9ca3af] font-mono">{sku}</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-[11px] uppercase tracking-wider text-[#6b7280] border-b border-[#F3F4F6]">
              <th className="px-3 py-2 text-left">Resolution</th>
              {allBatch.map(b => (
                <th key={b} className="px-3 py-2 text-right font-mono normal-case">batch={b}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {allRes.map(r => (
              <tr key={r} className="border-b border-[#F3F4F6] last:border-b-0">
                <td className="px-3 py-2 font-mono text-[#374151]">{r}px</td>
                {allBatch.map(b => {
                  const run = cell.get(`${r}|${b}`)
                  return <MatrixCell key={b} run={run} />
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MatrixCell({ run }: { run: RunState | undefined }) {
  if (!run) {
    return <td className="px-3 py-2 text-right text-[#d1d5db] text-xs">—</td>
  }
  const samplesPerSec = run.rate != null ? run.rate * run.batchSize : null
  return (
    <td className="px-3 py-2 text-right">
      {run.rate != null ? (
        <Link href={`/jobs/${run.jobId}`} className="block hover:bg-[#F7F4FC] rounded -mx-2 px-2 py-0.5">
          <div className="font-mono text-sm font-semibold text-[#16A34A]">
            {run.rate.toFixed(2)}<span className="text-[10px] text-[#9ca3af] font-normal"> step/s</span>
          </div>
          <div className="font-mono text-[10px] text-[#7E3AF2]">
            {samplesPerSec!.toFixed(1)} samp/s
          </div>
        </Link>
      ) : run.status === 'failed' ? (
        <Link href={`/jobs/${run.jobId}`} className="block hover:bg-[#FEF2F2] rounded -mx-2 px-2 py-0.5" title={run.errorMsg ?? 'failed'}>
          <span className="text-xs text-[#EF4444] font-semibold">
            {(run.errorMsg ?? '').toLowerCase().includes('memory') || (run.errorMsg ?? '').toLowerCase().includes('oom')
              ? 'OOM' : 'Failed'}
          </span>
        </Link>
      ) : run.status === 'closed' ? (
        <span className="text-xs text-[#9ca3af]">no rate</span>
      ) : (
        <Link href={`/jobs/${run.jobId}`} className="block hover:bg-[#F9FAFB] rounded -mx-2 px-2 py-0.5">
          <span className="text-xs text-[#6b7280]">running…</span>
        </Link>
      )}
    </td>
  )
}

/**
 * Side-effects-only component: opens the SSE log stream, polls the job
 * status, and dispatches updates to the shared `runs` state. Doesn't
 * render anything itself — the matrix view above reads from `runs`
 * directly. Kept as a separate component (instead of inline in
 * GpuMatrix) so each run gets its own effect lifecycle keyed on jobId.
 */
function RunSideEffects({ run, setRuns }: { run: RunState; setRuns: (f: (r: RunState[]) => RunState[]) => void }) {
  // Reuse the original RunRow's effects.
  return <RunRow run={run} setRuns={setRuns} headless />
}

function RunRow({ run, setRuns, headless = false }: { run: RunState; setRuns: (f: (r: RunState[]) => RunState[]) => void; headless?: boolean }) {
  const [tick, setTick] = useState(0)

  // Tick clock for the "elapsed" cell while the run is open
  useEffect(() => {
    if (headless) return  // headless instances don't render the elapsed cell
    if (run.status === 'measured' || run.status === 'failed' || run.status === 'closed') return
    const id = window.setInterval(() => setTick(t => t + 1), 1000)
    return () => window.clearInterval(id)
  }, [run.status, headless])

  // Open the SSE log stream and grep STEP_RATE. Once we find it (or the run
  // ends), close the stream so we're not holding open EventSources forever.
  useEffect(() => {
    if (run.status !== 'streaming') return
    const url = `/api/spark/jobs/${run.jobId}/logs`
    const es = new EventSource(url)

    const handle = (ev: MessageEvent) => {
      let line = ''
      try {
        const parsed = JSON.parse(ev.data)
        line = typeof parsed?.line === 'string' ? parsed.line : String(ev.data)
      } catch { line = String(ev.data) }
      const m = line.match(STEP_RATE_RE)
      if (m) {
        const rate = parseFloat(m[1])
        if (!Number.isNaN(rate)) {
          setRuns(prev => prev.map(r =>
            r.jobId === run.jobId
              ? { ...r, status: 'measured', rate, rateLineRaw: line }
              : r))
          es.close()
        }
      }
    }
    es.addEventListener('log', handle as EventListener)
    es.addEventListener('message', handle as EventListener)
    es.addEventListener('error', () => {
      // SSE auto-reconnects; do nothing here so we don't bail before the
      // STEP_RATE line arrives. The poll-status effect below catches
      // genuinely-failed runs.
    })

    return () => { es.close() }
  }, [run.jobId, run.status, setRuns])

  // Cheap status poll so we know if the job died before logging STEP_RATE.
  // Spark uses `succeeded` (not `completed`) as its terminal-success status.
  useEffect(() => {
    if (run.status !== 'streaming') return
    const id = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/spark/jobs/${run.jobId}`, { cache: 'no-store' })
        if (!res.ok) return
        const data = await res.json()
        const j = data?.job
        if (!j) return
        if (j.status === 'failed' || j.status === 'cancelled') {
          setRuns(prev => prev.map(r =>
            r.jobId === run.jobId
              ? { ...r, status: 'failed', errorMsg: j.error_message ?? `job ${j.status}` }
              : r))
        }
        if ((j.status === 'completed' || j.status === 'succeeded') && run.rate == null) {
          // Edge case: terminal-success without us catching STEP_RATE (e.g.
          // page refresh, slow log proxy). Mark as closed so the matrix
          // cell shows "no rate" rather than spinning forever.
          setRuns(prev => prev.map(r =>
            r.jobId === run.jobId
              ? { ...r, status: r.rate == null ? 'closed' : r.status }
              : r))
        }
      } catch { /* ignore poll errors */ }
    }, 5000)
    return () => window.clearInterval(id)
  }, [run.jobId, run.status, run.rate, setRuns])

  const elapsed = formatElapsed(Date.now() - run.startedAt)
  void tick   // ensure re-render

  // headless mode: only side-effects, no DOM. Used by the matrix view in the
  // 3-axis benchmark — that view renders its own cells from `runs`, but
  // still wants this component's effects (SSE stream + status poll) keyed
  // per jobId.
  if (headless) return null

  return (
    <tr className="border-b border-[#F3F4F6]">
      <td className="px-3 py-2">
        <p className="font-medium text-[#111827]">{run.label}</p>
        <p className="text-[11px] text-[#9ca3af] font-mono">{run.sku}</p>
      </td>
      <td className="px-3 py-2">
        <Link href={`/jobs/${run.jobId}`} className="text-xs font-mono text-[#7E3AF2] hover:underline">
          {run.jobId.slice(0, 8)}
        </Link>
      </td>
      <td className="px-3 py-2">
        <StatusPill status={run.status} />
        {run.errorMsg && <p className="text-[11px] text-[#EF4444] mt-0.5">{run.errorMsg}</p>}
      </td>
      <td className="px-3 py-2 text-right font-mono text-sm">
        {run.rate != null
          ? <span className="text-[#16A34A] font-semibold">{run.rate.toFixed(2)}</span>
          : run.status === 'closed' ? <span className="text-[#9ca3af]">—</span>
          : <span className="text-[#9ca3af]">…</span>}
      </td>
      <td className="px-3 py-2 text-right font-mono text-xs text-[#6b7280]">
        {elapsed}
      </td>
      <td className="px-3 py-2 text-right">
        <Link
          href={`/jobs/${run.jobId}`}
          className="text-[11px] text-[#6b7280] hover:text-[#7E3AF2]"
        >
          View →
        </Link>
      </td>
    </tr>
  )
}

function StatusPill({ status }: { status: RunState['status'] }) {
  const style = (
    status === 'measured'  ? 'bg-[#F0FDF4] text-[#16A34A]' :
    status === 'failed'    ? 'bg-[#FEF2F2] text-[#EF4444]' :
    status === 'streaming' ? 'bg-[#EFF6FF] text-[#1c64f2]' :
    status === 'closed'    ? 'bg-[#F9FAFB] text-[#9ca3af]' :
                             'bg-[#F9FAFB] text-[#6b7280]'
  )
  const label = (
    status === 'measured'  ? 'Measured' :
    status === 'failed'    ? 'Failed' :
    status === 'streaming' ? 'Running' :
    status === 'closed'    ? 'No rate' :
                             'Queued'
  )
  return <span className={`inline-block px-2 py-0.5 rounded-full text-[11px] font-medium ${style}`}>{label}</span>
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard refused */ }
  }
  return (
    <div className="relative">
      <pre className="bg-[#0f172a] text-[#e2e8f0] text-[12px] font-mono rounded-lg p-4 overflow-x-auto whitespace-pre">
        {code}
      </pre>
      <button
        type="button"
        onClick={onCopy}
        className="absolute top-2 right-2 px-2 py-1 text-[11px] bg-[#1e293b] hover:bg-[#334155] text-[#e2e8f0] rounded"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rs = s % 60
  return `${m}m ${rs}s`
}

function formatBytes(n: number): string {
  if (n < 1024)               return `${n} B`
  if (n < 1024 * 1024)        return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

/**
 * Emit a TS function body with measured rates inline. Failed measurements
 * keep the existing fallback so the function stays valid for unmeasured
 * SKUs (callers won't crash if calibration is partial).
 */
/**
 * Emit calibration code from a 3D matrix of measurements.
 *
 * Two outputs in one block:
 *
 *  1. `baselineStepsPerSec()` — uses the **reference cell** (512px, batch=2)
 *     for each GPU, since that's what `settingsMultiplier()` in
 *     spark-presets.ts treats as 1.0×. Falls back to the measured cell
 *     closest to the reference if the reference cell wasn't run or OOMed.
 *
 *  2. A `BENCHMARK_MEASUREMENTS` JSON literal with the full surface — every
 *     (gpu, res, batch) → step/sec — for archival, fitting a real
 *     `resolutionPenalty()` / `batchSizeMultiplier()`, or sanity-checking
 *     the existing heuristics. Doesn't get used at runtime; copy-paste it
 *     into a comment or commit it next to the constants.
 */
function emitTsBaseline(runs: RunState[]): string {
  // Pick the per-GPU reference rate: prefer (512, 2). If that's missing or
  // didn't measure, fall back to the cell with the smallest |res-512| +
  // |batch-2| weighted distance.
  const refRate = (gpuKey: string): { rate: number; res: number; batch: number } | null => {
    const cells = runs.filter(r => r.gpuKey === gpuKey && r.rate != null)
    if (cells.length === 0) return null
    const score = (r: RunState) =>
      Math.abs(r.resolution - 512) / 512 + Math.abs(r.batchSize - 2) * 0.5
    cells.sort((a, b) => score(a) - score(b))
    const c = cells[0]
    return { rate: c.rate!, res: c.resolution, batch: c.batchSize }
  }

  const date = new Date().toISOString().slice(0, 10)
  const fmt = (k: string, comment: string, fallback: number) => {
    const ref = refRate(k)
    if (!ref) return `  // ${k}: no successful cells this run (was ${fallback})`
    const note = ref.res === 512 && ref.batch === 2
      ? `reference cell`
      : `from ${ref.res}px/bs${ref.batch} — no 512px/bs2 cell`
    return `  if (sku.startsWith(${skuPrefix(k)})) return ${ref.rate.toFixed(2)}  // ${comment} (${note}, ${date}, was ${fallback})`
  }

  // Compact full-matrix dump for archival / future curve fitting.
  const matrix: Record<string, Record<string, number | string>> = {}
  for (const r of runs) {
    if (!matrix[r.gpuKey]) matrix[r.gpuKey] = {}
    const key = `${r.resolution}px/bs${r.batchSize}`
    matrix[r.gpuKey][key] = r.rate != null
      ? Number(r.rate.toFixed(3))
      : (r.status === 'failed' ? (r.errorMsg ?? 'failed') : r.status)
  }

  const lines: string[] = []
  lines.push('function baselineStepsPerSec(sku: string): number {')
  lines.push(fmt('t4',         'T4',           2))
  lines.push(fmt('l4',         'L4',           5))
  lines.push(fmt('a10',        'A10',          4))
  lines.push(fmt('l40s',       'L40S',         8))
  lines.push(fmt('rtxpro6000', 'RTX PRO 6000', 12))
  lines.push('  return 4')
  lines.push('}')
  lines.push('')
  lines.push(`// Full 3-axis sweep (step/sec). Use to fit resolutionPenalty()`)
  lines.push(`// and batchSizeMultiplier() instead of the current heuristics.`)
  lines.push(`const BENCHMARK_MEASUREMENTS = ${JSON.stringify(matrix, null, 2)}`)
  return lines.join('\n')
}

function skuPrefix(gpuKey: string): string {
  switch (gpuKey) {
    case 't4':         return "'g4dn'"
    case 'a10':        return "'g5'"
    case 'l4':         return "'g6.'"
    case 'l40s':       return "'g6e'"
    case 'rtxpro6000': return "'g7e'"
    default:           return "'??'"
  }
}
