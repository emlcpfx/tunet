/**
 * Status badge for Spark Compute v1 jobs.
 *
 * Spark statuses: queued | provisioning | running | completed | failed | cancelled
 * (different from the legacy DbJob statuses in <StatusBadge> from ui/badge.tsx)
 */

type SparkStatus = 'queued' | 'provisioning' | 'running' | 'completed' | 'failed' | 'cancelled' | string

const CONFIG: Record<string, { label: string; dot: string; text: string; bg: string }> = {
  queued:       { label: 'Queued',       dot: '#1c64f2', text: '#1c64f2', bg: '#EFF6FF' },
  provisioning: { label: 'Provisioning', dot: '#1c64f2', text: '#1c64f2', bg: '#EFF6FF' },
  running:      { label: 'Running',      dot: '#16A34A', text: '#16A34A', bg: '#F0FDF4' },
  completed:    { label: 'Completed',    dot: '#7E3AF2', text: '#7E3AF2', bg: '#F7F4FC' },
  failed:       { label: 'Failed',       dot: '#EF4444', text: '#EF4444', bg: '#FEF2F2' },
  cancelled:    { label: 'Cancelled',    dot: '#9ca3af', text: '#6b7280', bg: '#F9FAFB' },
}

const FALLBACK = { label: 'Unknown', dot: '#9ca3af', text: '#6b7280', bg: '#F9FAFB' }

export function SparkStatusBadge({ status }: { status: SparkStatus }) {
  const cfg = CONFIG[status] ?? FALLBACK
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
      style={{ background: cfg.bg, color: cfg.text }}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${status === 'running' ? 'animate-pulse' : ''}`}
        style={{ background: cfg.dot }}
      />
      {cfg.label || status}
    </span>
  )
}
