'use client'

/**
 * Client wrapper around <JobsTable> for /jobs. Owns the polling so the
 * list refreshes after a bulk-cancel and stays current while users sit on
 * the page. Renders its own empty state so the table always reflects the
 * latest poll, not the server-fetched seed.
 */

import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import {
  type SparkJob, ACTIVE_STATUSES, derivedStatus, formatStarted,
} from '@/lib/spark-types'
import { JobsTable } from './jobs-table'
import { Button } from '@/components/ui/button'

interface JobsListLiveProps {
  initialJobs:  SparkJob[]
  /** Filter slug from the URL (?status=...) — applied client-side so updates after polling stay in sync */
  statusFilter: string
}

export function JobsListLive({ initialJobs, statusFilter }: JobsListLiveProps) {
  const [jobs, setJobs] = useState<SparkJob[]>(initialJobs)

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/spark/jobs', { cache: 'no-store' })
      if (!res.ok) return
      const data = await res.json()
      if (Array.isArray(data.jobs)) setJobs(data.jobs as SparkJob[])
    } catch {
      // silent — non-critical, the page still shows the last good list
    }
  }, [])

  useEffect(() => {
    const anyActive = jobs.some(j => ACTIVE_STATUSES.has(j.status))
    const interval = anyActive ? 5_000 : 20_000
    const id = window.setInterval(refresh, interval)
    return () => window.clearInterval(id)
  }, [jobs, refresh])

  const filtered = (() => {
    if (statusFilter === 'all')    return jobs
    if (statusFilter === 'active') return jobs.filter(j => ACTIVE_STATUSES.has(j.status))
    // 'completed' is our display label for what Spark calls 'succeeded'
    // (older API docs say 'completed'; in practice we always see 'succeeded').
    // Match either so the filter UI doesn't return 0 jobs.
    if (statusFilter === 'completed') {
      return jobs.filter(j => {
        const s = derivedStatus(j)
        return s === 'succeeded' || s === 'completed'
      })
    }
    return jobs.filter(j => derivedStatus(j) === statusFilter)
  })()

  if (filtered.length === 0) {
    return (
      <div className="bg-white border border-dashed border-[#D1D5DB] rounded-xl p-10 text-center">
        <p className="text-[#6b7280] text-sm mb-4">
          {statusFilter === 'all' ? 'No jobs yet' : `No ${statusFilter} jobs`}
        </p>
        {statusFilter === 'all' && (
          <Link href="/jobs/new"><Button variant="secondary">Launch your first job</Button></Link>
        )}
      </div>
    )
  }

  return <JobsTable jobs={filtered} startedFormat={formatStarted} onAfterAction={refresh} />
}
