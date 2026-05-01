'use client'

/**
 * Polling status badge for job-detail page header. Re-fetches the job every
 * 5s while in an active state so the badge transitions queued → provisioning
 * → running without a hard reload, and uses derivedStatus() to mask Spark's
 * lag in flipping `status` to "running" once the container is up.
 */

import { useEffect, useState } from 'react'
import { type SparkJob, ACTIVE_STATUSES, derivedStatus } from '@/lib/spark-types'
import { SparkStatusBadge } from './status-badge'

interface LiveStatusBadgeProps {
  initialJob: SparkJob
}

export function LiveStatusBadge({ initialJob }: LiveStatusBadgeProps) {
  const [job, setJob] = useState<SparkJob>(initialJob)

  useEffect(() => {
    if (!ACTIVE_STATUSES.has(job.status)) return
    const id = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/spark/jobs/${job.id}`, { cache: 'no-store' })
        if (!res.ok) return
        const data = await res.json()
        if (data.job) setJob(data.job as SparkJob)
      } catch {
        // silent — header badge is non-critical, the timeline shows poll errors
      }
    }, 5_000)
    return () => window.clearInterval(id)
  }, [job.id, job.status])

  return <SparkStatusBadge status={job.status} liveOverride={derivedStatus(job)} />
}
