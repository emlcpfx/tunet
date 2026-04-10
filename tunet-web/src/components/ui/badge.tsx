import type { JobStatus } from '@/types'

const STATUS_CONFIG: Record<JobStatus, { label: string; dot: string; text: string; bg: string }> = {
  pending:      { label: 'Pending',      dot: '#D97706', text: '#D97706', bg: '#FFFBEB' },
  provisioning: { label: 'Starting',     dot: '#1c64f2', text: '#1c64f2', bg: '#EFF6FF' },
  running:      { label: 'Running',      dot: '#16A34A', text: '#16A34A', bg: '#F0FDF4' },
  stopped:      { label: 'Stopped',      dot: '#6b7280', text: '#6b7280', bg: '#F9FAFB' },
  failed:       { label: 'Failed',       dot: '#EF4444', text: '#EF4444', bg: '#FEF2F2' },
  terminated:   { label: 'Terminated',   dot: '#6b7280', text: '#6b7280', bg: '#F9FAFB' },
}

export function StatusBadge({ status }: { status: JobStatus }) {
  const cfg = STATUS_CONFIG[status]
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
      style={{ background: cfg.bg, color: cfg.text }}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${status === 'running' ? 'animate-pulse' : ''}`}
        style={{ background: cfg.dot }}
      />
      {cfg.label}
    </span>
  )
}

interface TierBadgeProps {
  tier: 'standard' | 'recommended' | 'pro' | 'premium'
}

const TIER_CONFIG = {
  standard:    { label: 'Standard',    style: 'bg-[#F9FAFB] text-[#374151]' },
  recommended: { label: 'Recommended', style: 'bg-[#F7F4FC] text-[#ae69f4]' },
  pro:         { label: 'Pro',         style: 'bg-[#EFF6FF] text-[#1c64f2]' },
  premium:     { label: 'Premium',     style: 'bg-[#FFFBEB] text-[#D97706]' },
}

export function TierBadge({ tier }: TierBadgeProps) {
  const cfg = TIER_CONFIG[tier]
  return (
    <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${cfg.style}`}>
      {cfg.label}
    </span>
  )
}
