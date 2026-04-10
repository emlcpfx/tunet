'use client'
import { useEffect, useState } from 'react'
import { Card, CardTitle, MetricCard } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { CREDIT_PACKS, formatCredits, type DbBillingEvent } from '@/types'

export default function BillingPage() {
  const [balance, setBalance]   = useState(0)
  const [events, setEvents]     = useState<DbBillingEvent[]>([])
  const [loading, setLoading]   = useState(false)
  const [checkoutPack, setCheckoutPack] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/billing/balance').then(r => r.json()).then(d => setBalance(d.balance_cents ?? 0))
    fetch('/api/billing/events').then(r => r.json()).then(d => setEvents(d.events ?? []))
  }, [])

  async function handleTopUp(packId: string, priceCents: number, balanceCents: number) {
    setCheckoutPack(packId)
    setLoading(true)
    const res = await fetch('/api/billing/checkout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ packId, priceCents, balanceCents }),
    })
    const { url } = await res.json()
    if (url) window.location.href = url
    else setLoading(false)
  }

  const monthlySpend = events
    .filter(e => e.type === 'compute_charge' && new Date(e.created_at) > new Date(Date.now() - 30 * 86400_000))
    .reduce((s, e) => s + Math.abs(e.amount_cents), 0)

  return (
    <div className="space-y-6 animate-slide-in max-w-3xl">
      <h1 className="text-2xl font-bold text-[#111827]">Billing</h1>

      {/* Balance + spend */}
      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          label="Credit Balance"
          value={formatCredits(balance)}
          sub="Available to spend"
          accent
        />
        <MetricCard
          label="Spend This Month"
          value={formatCredits(monthlySpend)}
          sub="Compute charges"
        />
      </div>

      {/* Top-up */}
      <Card>
        <CardTitle className="mb-4">Add Credits</CardTitle>
        <p className="text-sm text-[#6b7280] mb-5">
          Credits are charged per GPU-hour. No subscription, no expiry.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {CREDIT_PACKS.map(pack => (
            <button
              key={pack.id}
              onClick={() => handleTopUp(pack.id, pack.price_cents, pack.balance_cents)}
              disabled={loading}
              className="relative flex flex-col items-center p-4 border-2 border-[#e5e7eb] rounded-xl hover:border-[#ae69f4] hover:bg-[#F7F4FC] transition-all duration-150 disabled:opacity-50"
            >
              {pack.bonus_pct && (
                <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 bg-[#ae69f4] text-white text-xs font-semibold px-2 py-0.5 rounded-full">
                  +{pack.bonus_pct}%
                </span>
              )}
              <span className="text-2xl font-bold text-[#111827]">{pack.label}</span>
              <span className="text-xs text-[#6b7280] mt-1">{formatCredits(pack.balance_cents)} added to balance</span>
              {checkoutPack === pack.id && loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/80 rounded-xl">
                  <div className="w-5 h-5 border-2 border-[#ae69f4] border-t-transparent rounded-full animate-spin" />
                </div>
              )}
            </button>
          ))}
        </div>
        <p className="text-xs text-[#9ca3af] mt-3">Powered by Stripe. Secure payment.</p>
      </Card>

      {/* Pricing reference */}
      <Card>
        <CardTitle className="mb-3">GPU Pricing</CardTitle>
        <PricingTable />
      </Card>

      {/* Transaction history */}
      <Card>
        <CardTitle className="mb-4">Transaction History</CardTitle>
        {events.length === 0 ? (
          <p className="text-sm text-[#9ca3af]">No transactions yet</p>
        ) : (
          <div className="space-y-0">
            {events.slice(0, 50).map((evt, i) => (
              <div
                key={evt.id}
                className={`flex items-center justify-between py-3 ${i > 0 ? 'border-t border-[#F3F4F6]' : ''}`}
              >
                <div>
                  <p className="text-sm text-[#374151]">{evt.description ?? evt.type}</p>
                  <p className="text-xs text-[#9ca3af]">
                    {new Date(evt.created_at).toLocaleString('en-US', {
                      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                    })}
                  </p>
                </div>
                <span className={`text-sm font-semibold ${evt.amount_cents >= 0 ? 'text-[#16A34A]' : 'text-[#374151]'}`}>
                  {evt.amount_cents >= 0 ? '+' : ''}{formatCredits(evt.amount_cents)}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

function PricingTable() {
  const [gpus, setGpus] = useState<Array<{ display_name: string; vram_gb: number; platform_cost_per_hr: number; tier: string }>>([])

  useEffect(() => {
    fetch('/api/gpu-pricing').then(r => r.json()).then(setGpus)
  }, [])

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#e5e7eb]">
            <th className="text-left py-2 text-xs font-semibold text-[#374151]">GPU</th>
            <th className="text-left py-2 text-xs font-semibold text-[#374151]">VRAM</th>
            <th className="text-right py-2 text-xs font-semibold text-[#374151]">$/hr</th>
            <th className="text-right py-2 text-xs font-semibold text-[#374151]">$/4hr</th>
            <th className="text-right py-2 text-xs font-semibold text-[#374151]">$/8hr</th>
          </tr>
        </thead>
        <tbody>
          {gpus.map((gpu, i) => (
            <tr key={i} className={`${i > 0 ? 'border-t border-[#F3F4F6]' : ''}`}>
              <td className="py-2 font-medium text-[#111827]">{gpu.display_name}</td>
              <td className="py-2 text-[#6b7280]">{gpu.vram_gb} GB</td>
              <td className="py-2 text-right text-[#374151]">${gpu.platform_cost_per_hr.toFixed(2)}</td>
              <td className="py-2 text-right text-[#374151]">${(gpu.platform_cost_per_hr * 4).toFixed(2)}</td>
              <td className="py-2 text-right text-[#374151]">${(gpu.platform_cost_per_hr * 8).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
