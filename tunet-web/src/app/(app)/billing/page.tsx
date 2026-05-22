'use client'

/**
 * /billing — compute spend, usage, and live GPU pricing, all from Spark.
 *
 * Spark bills the Spark account directly (automatic withdrawals), so there is
 * NO TuNet credit wallet or Stripe checkout here — that was a holdover from the
 * earlier RunPod-reseller model. Each job carries
 * `total_attempted_compute_cost_usd_estimate` (a pre-billing estimate, not the
 * invoiced amount — Spark Fuse v1.17 §13.5.2), and the GPU rate table reads
 * live per-SKU prices from Spark's /estimate endpoint via /api/spark/pricing.
 *
 * Note: the Spark account is currently shared, so jobs/spend reflect
 * account-wide activity, not just this user's. Per-user attribution needs
 * per-user Spark auth (the SSO-to-Spark passthrough).
 */

import { useEffect, useMemo, useState } from 'react'
import { Card, CardTitle, MetricCard } from '@/components/ui/card'
import {
  type SparkJob,
  jobLabel,
  jobGpuDisplay,
  jobRuntimeMs,
  formatRuntime,
  formatStarted,
} from '@/lib/spark-types'

interface PricingEntry {
  instantUsdPerHr: number | null
  smartUsdPerHr: number | null
}
type PricingMap = Record<string, PricingEntry>

// GPU rate table — SKUs mirror /api/spark/pricing (which quotes live rates).
const GPU_ROWS: { sku: string; name: string; vram: number }[] = [
  { sku: 'g4dn.xlarge', name: 'NVIDIA T4', vram: 16 },
  { sku: 'g5.xlarge', name: 'NVIDIA A10', vram: 24 },
  { sku: 'g6.2xlarge', name: 'NVIDIA L4', vram: 24 },
  { sku: 'g6e.8xlarge', name: 'NVIDIA L40S', vram: 48 },
  { sku: 'g7e.2xlarge', name: 'NVIDIA RTX PRO 6000', vram: 96 },
]

function jobCostUsd(j: SparkJob): number | null {
  const raw = j.total_attempted_compute_cost_usd_estimate
  if (!raw) return null
  const n = parseFloat(raw)
  return Number.isFinite(n) ? n : null
}

function formatUsd(n: number): string {
  return `$${n.toFixed(2)}`
}

export default function DemoBillingPage() {
  const [jobs, setJobs] = useState<SparkJob[]>([])
  const [pricing, setPricing] = useState<PricingMap>({})
  const [usageError, setUsageError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/spark/jobs', { cache: 'no-store' })
      .then((r) => r.json())
      .then((d) => {
        if (Array.isArray(d.jobs)) setJobs(d.jobs as SparkJob[])
        else if (d.error) setUsageError(String(d.error))
      })
      .catch((e) => setUsageError(e instanceof Error ? e.message : 'Failed to load usage'))

    fetch('/api/spark/pricing')
      .then((r) => r.json())
      .then((d) => setPricing(d as PricingMap))
      .catch(() => {})
  }, [])

  const { spendMonthUsd, spendTotalUsd, usageRows } = useMemo(() => {
    const cutoff = Date.now() - 30 * 86_400_000
    let month = 0
    let total = 0
    const rows: { job: SparkJob; cost: number }[] = []
    for (const j of jobs) {
      const cost = jobCostUsd(j)
      if (cost === null) continue
      total += cost
      const endTs = j.terminal_at ?? j.started_running_at ?? j.created_at
      const endMs = endTs ? Date.parse(endTs) : NaN
      if (!Number.isNaN(endMs) && endMs > cutoff) month += cost
      rows.push({ job: j, cost })
    }
    rows.sort((a, b) => {
      const at = Date.parse(a.job.terminal_at ?? a.job.created_at ?? '') || 0
      const bt = Date.parse(b.job.terminal_at ?? b.job.created_at ?? '') || 0
      return bt - at
    })
    return { spendMonthUsd: month, spendTotalUsd: total, usageRows: rows }
  }, [jobs])

  return (
    <div className="space-y-6 animate-slide-in max-w-3xl">
      <h1 className="text-2xl font-bold text-[#111827]">Billing</h1>

      {/* Compute spend — from Spark per-job estimates */}
      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          label="Compute Spend (30d)"
          value={formatUsd(spendMonthUsd)}
          sub="Spark compute, last 30 days"
          accent
        />
        <MetricCard
          label="Compute Spend (all time)"
          value={formatUsd(spendTotalUsd)}
          sub="Across this Spark account"
        />
      </div>

      {/* Live GPU pricing from Spark */}
      <Card>
        <CardTitle className="mb-3">GPU Pricing</CardTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#e5e7eb]">
                <th className="text-left py-2 text-xs font-semibold text-[#374151]">GPU</th>
                <th className="text-left py-2 text-xs font-semibold text-[#374151]">VRAM</th>
                <th className="text-right py-2 text-xs font-semibold text-[#374151]">$/hr</th>
                <th className="text-right py-2 text-xs font-semibold text-[#374151]">$/hr (smart)</th>
              </tr>
            </thead>
            <tbody>
              {GPU_ROWS.map((gpu, i) => {
                const rate = pricing[gpu.sku]
                return (
                  <tr key={gpu.sku} className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''}`}>
                    <td className="py-2 font-medium text-[#111827]">{gpu.name}</td>
                    <td className="py-2 text-[#6b7280]">{gpu.vram} GB</td>
                    <td className="py-2 text-right text-[#374151]">
                      {rate?.instantUsdPerHr != null ? `$${rate.instantUsdPerHr.toFixed(2)}` : '—'}
                    </td>
                    <td className="py-2 text-right text-[#374151]">
                      {rate?.smartUsdPerHr != null ? `$${rate.smartUsdPerHr.toFixed(2)}` : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-[#9ca3af] mt-3">
          Live rates from Spark. Smart-mode is preemptible spare capacity — cheaper, may be interrupted.
        </p>
      </Card>

      {/* Compute usage from Spark */}
      <Card>
        <CardTitle className="mb-4">Compute Usage</CardTitle>
        {usageError ? (
          <p className="text-sm text-[#EF4444]">Couldn&apos;t load usage: {usageError}</p>
        ) : usageRows.length === 0 ? (
          <p className="text-sm text-[#9ca3af]">No compute usage yet</p>
        ) : (
          <div className="space-y-0">
            {usageRows.slice(0, 50).map(({ job, cost }, i) => (
              <div
                key={job.id}
                className={`flex items-center justify-between py-3 ${i > 0 ? 'border-t border-[#F3F4F6]' : ''}`}
              >
                <div className="min-w-0">
                  <p className="text-sm text-[#374151] truncate">{jobLabel(job)}</p>
                  <p className="text-xs text-[#9ca3af]">
                    {jobGpuDisplay(job)} · {formatRuntime(jobRuntimeMs(job))} · {formatStarted(job)}
                  </p>
                </div>
                <span className="text-sm font-semibold text-[#374151] flex-shrink-0 ml-3">
                  {formatUsd(cost)}
                </span>
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-[#9ca3af] mt-3">
          Compute costs are Spark Fuse pre-billing estimates and reflect account-wide activity.
        </p>
      </Card>
    </div>
  )
}
