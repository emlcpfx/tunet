/**
 * Periodic training-alert poller.
 *
 * Runs on Vercel Cron (configured in vercel.json). For every active Spark
 * job, fetches the training.log, runs analyze_training-equivalent heuristics,
 * and emails the user when the run hits a notable transition:
 *
 *   • plateau       — best loss is 30+ epochs old (slow ramp-down)
 *   • training_done — best loss is 50+ epochs old (almost certainly done)
 *   • diverging     — loss is climbing > 1%/epoch (LR / data problem)
 *
 * Per (job, kind) cooldown of 4h via job_alerts table — no spam at 31, 32, 33.
 *
 * The alert prefs + recipient email live in job env at submit time
 *   TUNET_ALERT_EMAIL=user@example.com
 *   TUNET_ALERT_PLATEAU=1            (opt in to plateau + training_done)
 *   TUNET_ALERT_DIVERGING=1
 *
 * Missing env = user opted out for that job. We skip silently.
 */

import { NextResponse } from 'next/server'
import { listJobs, type SparkJob } from '@/lib/spark'
import { ACTIVE_STATUSES } from '@/lib/spark-types'
import { analyzeJobForAlerts, type AlertKind } from '@/lib/training-alerts'
import { sendEmail, plateauEmail, trainingDoneEmail, divergingEmail, type JobContext } from '@/lib/email'
import { createServiceClient } from '@/lib/supabase'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 60

const COOLDOWN_MS = 4 * 60 * 60 * 1000  // 4h between alerts of the same kind

interface AlertRow {
  job_id: string
  kind:   string
  fired_at: string
}

export async function GET(req: Request) {
  // Same auth scheme as billing-tick — Vercel injects CRON_SECRET as Bearer.
  const auth = req.headers.get('authorization')
  if (auth !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  let jobs: SparkJob[]
  try {
    jobs = await listJobs()
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : 'list jobs failed' },
      { status: 502 },
    )
  }

  const active = jobs.filter(j => ACTIVE_STATUSES.has(j.status))
  if (active.length === 0) {
    return NextResponse.json({ ok: true, scanned: 0, fired: 0, skipped: 0 })
  }

  // Bulk-load recent dedup rows so we don't hit the DB per (job, kind) pair.
  const svc = createServiceClient()
  const since = new Date(Date.now() - COOLDOWN_MS).toISOString()
  const { data: recentRaw } = await svc
    .from('job_alerts')
    .select('job_id, kind, fired_at')
    .in('job_id', active.map(j => j.id))
    .gte('fired_at', since)

  const recent = new Map<string, Set<string>>()  // jobId → set of kinds
  for (const r of (recentRaw ?? []) as AlertRow[]) {
    if (!recent.has(r.job_id)) recent.set(r.job_id, new Set())
    recent.get(r.job_id)!.add(r.kind)
  }

  const stats = { scanned: 0, fired: 0, skipped_no_email: 0, skipped_no_log: 0, skipped_cooldown: 0 }
  const fires: { jobId: string; kind: AlertKind; ok: boolean; error?: string }[] = []

  for (const job of active) {
    stats.scanned += 1

    const email = job.env?.TUNET_ALERT_EMAIL?.trim()
    if (!email) { stats.skipped_no_email += 1; continue }

    const wantsPlateau   = job.env?.TUNET_ALERT_PLATEAU   === '1'
    const wantsDiverging = job.env?.TUNET_ALERT_DIVERGING === '1'
    if (!wantsPlateau && !wantsDiverging) { stats.skipped_no_email += 1; continue }

    const result = await analyzeJobForAlerts(job)
    if (!result.snapshot) { stats.skipped_no_log += 1; continue }

    const alreadyFired = recent.get(job.id) ?? new Set<string>()

    for (const kind of result.recommend) {
      // Match recommendation against the user's per-job opt-in
      if ((kind === 'plateau' || kind === 'training_done') && !wantsPlateau)   continue
      if (kind === 'diverging' && !wantsDiverging)                              continue
      if (alreadyFired.has(kind))                                              { stats.skipped_cooldown += 1; continue }

      const ctx: JobContext = {
        jobId:        job.id,
        jobName:      job.env?.TUNET_JOB_NAME ?? job.id.slice(0, 8),
        jobUrl:       buildJobUrl(job.id),
        currentEpoch: result.snapshot.currentEpoch.toFixed(1),
        bestLoss:     result.snapshot.bestLoss.toFixed(5),
        bestEpoch:    result.snapshot.bestEpoch.toFixed(1),
        status:       result.snapshot.statusLabel,
      }

      const tmpl =
        kind === 'plateau'       ? plateauEmail(ctx) :
        kind === 'training_done' ? trainingDoneEmail(ctx) :
                                   divergingEmail(ctx)

      const send = await sendEmail({ to: email, subject: tmpl.subject, text: tmpl.text })
      fires.push({ jobId: job.id, kind, ok: send.ok, error: send.error })

      if (send.ok || send.skipped) {
        // Persist the dedup row even on dry-run (skipped) so the local dev
        // logs don't repeat. On real send failure we DON'T persist — let
        // the next tick try again.
        await svc.from('job_alerts').insert({
          job_id: job.id,
          kind,
          email,
          meta: {
            current_epoch: result.snapshot.currentEpoch,
            best_loss:     result.snapshot.bestLoss,
            best_epoch:    result.snapshot.bestEpoch,
            rel_slope:     result.snapshot.relSlope,
            status_label:  result.snapshot.statusLabel,
          },
        })
        stats.fired += 1
      }
    }
  }

  return NextResponse.json({
    ok: true,
    timestamp: new Date().toISOString(),
    ...stats,
    fires,
  })
}

function buildJobUrl(id: string): string {
  const base = process.env.PUBLIC_APP_URL ?? 'http://localhost:3000'
  return `${base.replace(/\/+$/, '')}/demo/jobs/${id}`
}
