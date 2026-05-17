/**
 * Outbound transactional email — Resend.
 *
 * We use Resend instead of SMTP for two reasons:
 *   1. No long-lived TCP from the Vercel runtime (their cron / serverless
 *      functions get killed mid-connection).
 *   2. Free tier is generous (~3k emails/mo) and the SDK is one tiny POST.
 *
 * Required env (set in tunet-web/.env.local):
 *   RESEND_API_KEY=re_...
 *   RESEND_FROM=alerts@yourdomain.com    (must be a verified sender)
 *   PUBLIC_APP_URL=https://...           (for links back to the job page)
 */

import 'server-only'
import { Resend } from 'resend'

let _resend: Resend | null = null

function client(): Resend | null {
  if (_resend) return _resend
  const key = process.env.RESEND_API_KEY
  if (!key) return null
  _resend = new Resend(key)
  return _resend
}

export interface SendArgs {
  to:       string
  subject:  string
  /** Plain-text body — we keep emails plain by default for dev clarity. */
  text:     string
  /** Optional HTML body — falls back to text if absent. */
  html?:    string
}

export interface SendResult {
  ok:        boolean
  messageId?: string
  error?:    string
  /** True if RESEND_API_KEY was missing — we treat this as a soft skip,
   *  not an error, so dev environments don't get noisy. */
  skipped?:  boolean
}

export async function sendEmail(args: SendArgs): Promise<SendResult> {
  const c = client()
  if (!c) {
    console.warn(`[email] RESEND_API_KEY not set — would have sent: "${args.subject}" → ${args.to}`)
    return { ok: false, skipped: true }
  }
  const from = process.env.RESEND_FROM
  if (!from) {
    console.error('[email] RESEND_FROM not set')
    return { ok: false, error: 'RESEND_FROM not configured' }
  }

  try {
    const res = await c.emails.send({
      from,
      to:      args.to,
      subject: args.subject,
      text:    args.text,
      html:    args.html ?? plainToHtml(args.text),
    })
    if (res.error) return { ok: false, error: res.error.message }
    return { ok: true, messageId: res.data?.id }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : 'send failed' }
  }
}

/** Minimal text → HTML so the HTML version isn't worse than the text one. */
function plainToHtml(text: string): string {
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  return `<pre style="font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap;">${escaped}</pre>`
}

// ── Templates ────────────────────────────────────────────────────────────────

export interface JobContext {
  jobId:        string
  jobName:      string
  jobUrl:       string
  /** Pretty epoch number e.g. "ep 42.5" */
  currentEpoch: string
  bestLoss:     string
  bestEpoch:    string
  /** "Best 31ep ago" or "Slope -0.7%" — short status from analyze_training */
  status:       string
}

export function plateauEmail(ctx: JobContext): { subject: string; text: string } {
  return {
    subject: `[tunet] ${ctx.jobName}: training has plateaued`,
    text:
`Your training run "${ctx.jobName}" hasn't improved in a while.

Status:        ${ctx.status}
Current epoch: ${ctx.currentEpoch}
Best loss:     ${ctx.bestLoss} (at epoch ${ctx.bestEpoch})

The model isn't getting better — every additional epoch is costing money
without improving the result. You can stop the job to save credit:

    ${ctx.jobUrl}

If this looks fine to you (sometimes loss looks flat but the model is still
refining detail), feel free to keep it running.

— TuNet`,
  }
}

export function trainingDoneEmail(ctx: JobContext): { subject: string; text: string } {
  return {
    subject: `[tunet] ${ctx.jobName}: training likely done — consider stopping`,
    text:
`Your training run "${ctx.jobName}" looks done.

Status:        ${ctx.status}
Current epoch: ${ctx.currentEpoch}
Best loss:     ${ctx.bestLoss} (at epoch ${ctx.bestEpoch})

You don't have auto-stop configured, so the job will keep running on the
GPU until max_steps. If the loss curve looks like it's done, stop now to
save the rest of your credit:

    ${ctx.jobUrl}

— TuNet`,
  }
}

export function divergingEmail(ctx: JobContext): { subject: string; text: string } {
  return {
    subject: `[tunet] ${ctx.jobName}: training appears to be diverging`,
    text:
`Your training run "${ctx.jobName}" loss is going UP, not down.

Status:        ${ctx.status}
Current epoch: ${ctx.currentEpoch}
Best loss:     ${ctx.bestLoss} (at epoch ${ctx.bestEpoch})

This usually means the learning rate is too high, the data has bad pairs,
or something is broken. You probably want to stop and investigate:

    ${ctx.jobUrl}

— TuNet`,
  }
}
