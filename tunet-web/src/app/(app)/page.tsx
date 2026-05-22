/**
 * / — live Spark Compute dashboard.
 *
 * Server component fetches the initial job list directly from Spark to avoid
 * first-paint flash, then hands off to <DashboardLive> which polls
 * /api/spark/jobs every 5s (active) / 20s (idle) so status, runtimes, and the
 * active card update without a hard reload.
 *
 * Layout mirrors Spark/frontend/dashboard.html: metric strip + active job
 * card + recent jobs grid + "New Job" CTA.
 */

import Link from 'next/link'
import { listJobs } from '@/lib/spark'
import type { SparkJob } from '@/lib/spark-types'
import { DashboardLive } from '@/components/spark/dashboard-live'
import { Button } from '@/components/ui/button'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export default async function DemoDashboard() {
  let jobs: SparkJob[] = []
  let fetchError: string | null = null

  try {
    jobs = await listJobs()
  } catch (e) {
    fetchError = e instanceof Error ? e.message : 'Failed to load jobs'
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#111827]">Dashboard</h1>
          <p className="text-sm text-[#6b7280] mt-1">
            Train image-to-image models on cloud GPUs. Pick a preset, point at frames, go.
          </p>
        </div>
        <Link href="/jobs/new">
          <Button size="md">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>
            </svg>
            New Job
          </Button>
        </Link>
      </div>

      {fetchError && (
        <div className="px-4 py-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
          <strong>Couldn&apos;t load Spark jobs:</strong> {fetchError}
          <p className="text-xs mt-1 text-[#6b7280]">
            Your session may have expired — try signing out and back in.
          </p>
        </div>
      )}

      <DashboardLive initialJobs={jobs} />
    </div>
  )
}
