/**
 * Status badge for Spark Compute v1 jobs.
 *
 * Spark statuses: queued | provisioning | running | completed | failed | cancelled
 * (different from the legacy DbJob statuses in <StatusBadge> from ui/badge.tsx)
 *
 * Pass `liveOverride` to force the displayed status (e.g. when the parent
 * already knows logs are streaming so the container must be running, even
 * though Spark's job record hasn't transitioned yet).
 */

type SparkStatus = 'queued' | 'provisioning' | 'running' | 'succeeded' | 'completed' | 'failed' | 'cancelled' | string

// Spark's API uses `succeeded` as its terminal-success status. Older docs
// say `completed` but every real job we've seen returns `succeeded`. We
// keep both keys so the UI doesn't fall through to "Unknown" if either
// shows up. Display the same — purple "Completed" — since users don't
// care about Spark's internal naming.
const CONFIG: Record<string, { label: string; dot: string; text: string; bg: string }> = {
  queued:       { label: 'Queued',       dot: '#1c64f2', text: '#1c64f2', bg: '#EFF6FF' },
  provisioning: { label: 'Provisioning', dot: '#1c64f2', text: '#1c64f2', bg: '#EFF6FF' },
  running:      { label: 'Running',      dot: '#16A34A', text: '#16A34A', bg: '#F0FDF4' },
  succeeded:    { label: 'Completed',    dot: '#7E3AF2', text: '#7E3AF2', bg: '#F7F4FC' },
  completed:    { label: 'Completed',    dot: '#7E3AF2', text: '#7E3AF2', bg: '#F7F4FC' },
  failed:       { label: 'Failed',       dot: '#EF4444', text: '#EF4444', bg: '#FEF2F2' },
  cancelled:    { label: 'Cancelled',    dot: '#9ca3af', text: '#6b7280', bg: '#F9FAFB' },
}

const FALLBACK = { label: 'Unknown', dot: '#9ca3af', text: '#6b7280', bg: '#F9FAFB' }

export function SparkStatusBadge({ status, liveOverride }: { status: SparkStatus; liveOverride?: SparkStatus }) {
  const shown = liveOverride ?? status
  const cfg = CONFIG[shown] ?? FALLBACK
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
      style={{ background: cfg.bg, color: cfg.text }}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${shown === 'running' ? 'animate-pulse' : ''}`}
        style={{ background: cfg.dot }}
      />
      {cfg.label || shown}
    </span>
  )
}
