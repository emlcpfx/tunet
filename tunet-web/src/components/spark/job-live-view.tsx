'use client'

/**
 * Wrapper that combines ProvisioningTimeline + LogStreamPanel and shares the
 * `hasLogs` flag between them. Once the first log line arrives, the timeline
 * auto-collapses to a one-line summary so the log panel can fill the view.
 */

import { useState } from 'react'
import { ProvisioningTimeline } from './provisioning-timeline'
import { LogStreamPanel }       from './log-stream-panel'
import type { SparkJob } from '@/lib/spark-types'

interface JobLiveViewProps {
  initialJob:    SparkJob
  initiallyLive: boolean
}

export function JobLiveView({ initialJob, initiallyLive }: JobLiveViewProps) {
  const [hasLogs, setHasLogs] = useState(false)
  const [timelineCollapsed, setTimelineCollapsed] = useState(false)

  return (
    <div className="space-y-4">
      {/* Timeline — collapses once logs are flowing */}
      {!timelineCollapsed ? (
        <ProvisioningTimeline
          jobId={initialJob.id}
          initialJob={initialJob}
          hasLogs={hasLogs}
          onLogsStart={() => {
            // Brief delay so the user sees the green "Running" check before collapse
            setTimeout(() => setTimelineCollapsed(true), 1200)
          }}
        />
      ) : (
        <button
          type="button"
          onClick={() => setTimelineCollapsed(false)}
          className="text-xs text-[#7E3AF2] hover:text-[#6C2BD9] flex items-center gap-1"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="9 18 15 12 9 6" style={{ transform: 'rotate(0deg)' }} />
          </svg>
          Show provisioning timeline
        </button>
      )}

      {/* Log stream */}
      <LogStreamPanel
        jobId={initialJob.id}
        initiallyLive={initiallyLive}
        onFirstLine={() => setHasLogs(true)}
      />
    </div>
  )
}
