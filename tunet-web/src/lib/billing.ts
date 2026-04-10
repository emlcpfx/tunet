import { createServiceClient } from './supabase'
import { stopPod, terminatePod, getPod } from './runpod'
import { autoStopThresholdCents } from './pricing'
import type { DbJob } from '@/types'

// ── Billing tick — called by Vercel Cron every 5 minutes ─────────────────────

export async function runBillingTick(): Promise<{ charged: number; stopped: number }> {
  const svc = createServiceClient()
  const now = new Date()
  let totalCharged = 0
  let totalStopped = 0

  // Fetch all running jobs
  const { data: jobs, error } = await svc
    .from('jobs')
    .select('*')
    .eq('status', 'running')

  if (error || !jobs) return { charged: 0, stopped: 0 }

  for (const job of jobs as DbJob[]) {
    if (!job.platform_cost_per_hr || !job.billing_last_tick_at) continue

    const lastTick = new Date(job.billing_last_tick_at)
    const elapsedMinutes = (now.getTime() - lastTick.getTime()) / 60_000

    // Charge for elapsed time
    const chargeCents = Math.ceil(
      (job.platform_cost_per_hr / 60) * elapsedMinutes * 100,
    )

    if (chargeCents <= 0) continue

    // Deduct from user balance and record event
    const { data: user } = await svc
      .from('users')
      .select('credit_balance_cents')
      .eq('id', job.user_id)
      .single()

    if (!user) continue

    const newBalance = user.credit_balance_cents - chargeCents

    await svc
      .from('users')
      .update({ credit_balance_cents: Math.max(0, newBalance) })
      .eq('id', job.user_id)

    await svc.from('billing_events').insert({
      user_id:     job.user_id,
      job_id:      job.id,
      type:        'compute_charge',
      amount_cents: -chargeCents,
      description: `Compute: ${job.gpu_display_name ?? job.gpu_type_id}`,
    })

    await svc
      .from('jobs')
      .update({
        accumulated_cost_cents: job.accumulated_cost_cents + chargeCents,
        billing_last_tick_at:   now.toISOString(),
      })
      .eq('id', job.id)

    totalCharged += chargeCents

    // Auto-stop if balance is critically low
    const threshold = autoStopThresholdCents(job.platform_cost_per_hr)
    if (newBalance < threshold && job.pod_id) {
      try {
        await stopPod(job.pod_id)
        await svc.from('jobs').update({ status: 'stopped' }).eq('id', job.id)
        totalStopped++
        // TODO: send low-balance email notification
      } catch {
        // Pod may already be stopped — ignore
      }
    }
  }

  // Reconcile: update status of pods that RunPod terminated externally
  await reconcileTerminatedPods()

  return { charged: totalCharged, stopped: totalStopped }
}

// ── Reconciliation — detect pods RunPod killed that we didn't ───────────────

async function reconcileTerminatedPods(): Promise<void> {
  const svc = createServiceClient()

  const { data: jobs } = await svc
    .from('jobs')
    .select('id, pod_id, status')
    .in('status', ['running', 'provisioning'])
    .not('pod_id', 'is', null)

  if (!jobs) return

  for (const job of jobs as Pick<DbJob, 'id' | 'pod_id' | 'status'>[]) {
    if (!job.pod_id) continue
    try {
      const pod = await getPod(job.pod_id)
      if (!pod || pod.desiredStatus === 'EXITED' || pod.desiredStatus === 'DEAD') {
        await svc
          .from('jobs')
          .update({ status: 'terminated', ended_at: new Date().toISOString() })
          .eq('id', job.id)
      } else if (pod.desiredStatus === 'RUNNING' && job.status === 'provisioning') {
        await svc
          .from('jobs')
          .update({
            status:       'running',
            started_at:   new Date().toISOString(),
            billing_last_tick_at: new Date().toISOString(),
            runpod_cost_per_hr: pod.costPerHr,
            gpu_display_name: pod.machine?.gpuDisplayName ?? job.pod_id,
          })
          .eq('id', job.id)
      }
    } catch {
      // RunPod API error — skip this pod
    }
  }
}

// ── Manual credit adjustment (admin) ─────────────────────────────────────────

export async function adjustCredits(
  userId: string,
  amountCents: number,
  description: string,
): Promise<void> {
  const svc = createServiceClient()

  await svc.rpc('adjust_credits', {
    p_user_id:     userId,
    p_amount_cents: amountCents,
    p_description:  description,
  })
}
