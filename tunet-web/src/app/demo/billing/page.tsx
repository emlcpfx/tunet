'use client'
import { Card, CardTitle, MetricCard } from '@/components/ui/card'
import { CREDIT_PACKS, formatCredits, type DbBillingEvent } from '@/types'

const MOCK_BALANCE = 5250
const MOCK_MONTHLY_SPEND = 851

const MOCK_EVENTS: DbBillingEvent[] = [
  { id: 'evt-1', user_id: 'demo', job_id: 'job-001', type: 'compute_charge', amount_cents: -135, description: 'dolly-shot-fix-v3 · 18m on L40S 48GB', stripe_payment_intent: null, created_at: new Date(Date.now() - 18 * 60 * 1000).toISOString() },
  { id: 'evt-2', user_id: 'demo', job_id: 'job-002', type: 'compute_charge', amount_cents: -716, description: 'hand-fix-v1 · 4h on A10 24GB', stripe_payment_intent: null, created_at: new Date(Date.now() - 4 * 3600 * 1000).toISOString() },
  { id: 'evt-3', user_id: 'demo', job_id: null,      type: 'top_up',         amount_cents: 5250, description: 'Credit top-up — $50 pack (+5%)', stripe_payment_intent: 'pi_demo123', created_at: new Date(Date.now() - 2 * 24 * 3600 * 1000).toISOString() },
  { id: 'evt-4', user_id: 'demo', job_id: 'job-004', type: 'compute_charge', amount_cents: -2100, description: 'paintout-LoRA-v2 · 4h on L40S Pro', stripe_payment_intent: null, created_at: new Date(Date.now() - 3 * 24 * 3600 * 1000 + 4 * 3600 * 1000).toISOString() },
]

const MOCK_GPUS = [
  { display_name: 'NVIDIA A10 24GB',  vram_gb: 24, platform_cost_per_hr: 1.19 },
  { display_name: 'NVIDIA L4 24GB',   vram_gb: 24, platform_cost_per_hr: 1.79 },
  { display_name: 'NVIDIA L40S 48GB', vram_gb: 48, platform_cost_per_hr: 4.50 },
  { display_name: 'NVIDIA L40S Pro',  vram_gb: 48, platform_cost_per_hr: 5.25 },
]

export default function DemoBillingPage() {
  return (
    <div className="space-y-6 animate-slide-in max-w-3xl">
      <h1 className="text-2xl font-bold text-[#111827]">Billing</h1>

      <div className="px-3 py-2 bg-[#FFFBEB] border border-[#FDE68A] rounded-lg text-xs text-[#D97706]">
        Demo mode — Stripe checkout is disabled.
      </div>

      <div className="grid grid-cols-2 gap-4">
        <MetricCard label="Credit Balance"   value={formatCredits(MOCK_BALANCE)}       sub="Available to spend" accent />
        <MetricCard label="Spend This Month" value={formatCredits(MOCK_MONTHLY_SPEND)} sub="Compute charges"          />
      </div>

      <Card>
        <CardTitle className="mb-4">Add Credits</CardTitle>
        <p className="text-sm text-[#6b7280] mb-5">Credits are charged per GPU-hour. No subscription, no expiry.</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {CREDIT_PACKS.map(pack => (
            <div
              key={pack.id}
              className="relative flex flex-col items-center p-4 border-2 border-[#e5e7eb] rounded-xl opacity-60 cursor-not-allowed"
            >
              {pack.bonus_pct && (
                <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 bg-[#ae69f4] text-white text-xs font-semibold px-2 py-0.5 rounded-full">
                  +{pack.bonus_pct}%
                </span>
              )}
              <span className="text-2xl font-bold text-[#111827]">{pack.label}</span>
              <span className="text-xs text-[#6b7280] mt-1">{formatCredits(pack.balance_cents)} added</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-[#9ca3af] mt-3">Powered by Stripe. Secure payment.</p>
      </Card>

      <Card>
        <CardTitle className="mb-3">GPU Pricing</CardTitle>
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
              {MOCK_GPUS.map((gpu, i) => (
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
      </Card>

      <Card>
        <CardTitle className="mb-4">Transaction History</CardTitle>
        <div className="space-y-0">
          {MOCK_EVENTS.map((evt, i) => (
            <div key={evt.id} className={`flex items-center justify-between py-3 ${i > 0 ? 'border-t border-[#F3F4F6]' : ''}`}>
              <div>
                <p className="text-sm text-[#374151]">{evt.description ?? evt.type}</p>
                <p className="text-xs text-[#9ca3af]">
                  {new Date(evt.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
              <span className={`text-sm font-semibold ${evt.amount_cents >= 0 ? 'text-[#16A34A]' : 'text-[#374151]'}`}>
                {evt.amount_cents >= 0 ? '+' : ''}{formatCredits(evt.amount_cents)}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
