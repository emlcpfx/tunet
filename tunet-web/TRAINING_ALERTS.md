# Training Alerts — Email Notifications for Stalled / Done / Diverging Runs

**Audience:** Walt, plus anyone picking up the web side.
**Author:** Eric
**Status:** Built, not yet enabled in prod (needs Resend key + DB migration).
**Date:** 2026-05-01

---

## What it does

When Eric submits a training job through `/demo/jobs/new`, he can opt into
email alerts. The web app then watches the job's `training.log` (streamed
to ShareSync by the Spark agent), runs the same heuristic the desktop app
uses to decide if a run is plateauing / done / diverging, and emails Eric
when one of those transitions fires.

The motivating use case: Spark bills per-second. If Eric forgets to set
auto-stop and his loss has flatlined for an hour, that's $1.65–$2.99 of
GPU time burnt for nothing. We catch it within 10 min of the heuristic
firing and tell him to stop the job.

Three alert kinds:

| Kind             | Trigger                                                              | Subject line                                |
|------------------|----------------------------------------------------------------------|---------------------------------------------|
| `plateau`        | Best loss is 30+ epochs old (and slope ≈ 0)                          | "training has plateaued"                    |
| `training_done`  | Best loss is 50+ epochs old                                          | "training likely done — consider stopping"  |
| `diverging`      | Smoothed slope > +1%/epoch (loss is going UP)                        | "training appears to be diverging"          |

`plateau` and `training_done` are mutually exclusive — when both qualify
on the same tick, only `training_done` fires (otherwise we'd send two
nearly-identical emails). `diverging` is urgent and short-circuits the
others.

---

## How the user opts in

Bottom of `/demo/jobs/new` → Step 4 → "Email alerts" row. Three controls:

- **Email field** — recipient (empty = opted out for this job)
- **Notify when training stalls / done** — covers plateau + training_done
- **Notify if loss starts going up** — covers diverging

Both checkboxes default to ON, but only fire if the email field has a value.
The whole thing is per-job — no global user-prefs table to manage. We
considered a global setting but decided the per-job model is better:

- Each run is a deliberate experiment. The user is *already* on the form
  when they decide.
- No "I disabled alerts last month for that one short run" surprises.
- No prefs migration when the user changes email providers.

The prefs travel as **job env vars**, set on submit in
`src/app/api/spark/training-jobs/route.ts`:

```
TUNET_ALERT_EMAIL=eric@cleanplatefx.com
TUNET_ALERT_PLATEAU=1            # opted in to plateau + training_done
TUNET_ALERT_DIVERGING=1
```

Spark echoes these back in the job detail response, so the cron poller
reads them straight off the job object — no separate lookup.

---

## How the cron decides what to fire

`src/app/api/cron/training-alerts/route.ts` runs every 10 minutes (Vercel
Cron, see `vercel.json`). For each active job:

1. **Skip if not opted in** — no `TUNET_ALERT_EMAIL`, or neither alert
   kind enabled → silent skip.
2. **Fetch `training.log`** from the job's output ShareSync dir via the
   existing `fetchOutputFile()` helper in `src/lib/spark.ts`. (Not a
   range request — the file is small, ~10 KB per 100 epochs. Re-pulling
   it every 10 min per active job is negligible vs. the GPU bill we're
   trying to save.)
3. **Parse `Epoch[N] Step[M] L1:0.x` lines** with the same regex the live
   chart uses (`src/components/spark/use-training-stream.ts`). Only the
   train series — val isn't needed for these heuristics.
4. **Run analyze** — `src/lib/training-alerts.ts` is a port of
   `training_monitor.py:649 analyze_training`. Same window (last
   `min(20, currentEpoch * 0.5)` epochs), same EMA (α=0.95), same linear
   regression slope, same thresholds. Returns a snapshot + an array of
   recommended alert kinds.
5. **Apply user opt-ins** — drop kinds the user didn't subscribe to.
6. **Apply 4h cooldown per (job, kind)** — bulk-fetched from the
   `job_alerts` table at the top of the tick.
7. **Send + persist** — render the template (`src/lib/email.ts`), POST to
   Resend, and on success insert a `job_alerts` row so we don't double-fire.

The cron is idempotent: if Resend fails, we don't persist, so the next tick
retries. If Resend isn't configured (no `RESEND_API_KEY`), we log
"would have sent" and *do* persist — that way local dev runs don't keep
firing the same alert every 10 min.

The same Bearer-secret pattern as `/api/cron/billing-tick` protects it:

```
GET /api/cron/training-alerts
Authorization: Bearer ${CRON_SECRET}
```

---

## Why we copied the heuristic verbatim from the desktop app

`training_monitor.py:649-743` (the `analyze_training` method) is what
shows the green/yellow/red status pills at the bottom of the desktop
training monitor. Eric has tuned those thresholds against his real
runs. If the email says "Plateau — may stop" but the desktop UI says
"Stable", we lose user trust immediately.

So `src/lib/training-alerts.ts` mirrors the Python line-for-line:

| Python (`training_monitor.py`)          | TS (`training-alerts.ts`)                     |
|------------------------------------------|------------------------------------------------|
| `window_epochs = min(20, current * 0.5)` | `Math.min(20, currentEpoch * 0.5)`            |
| `alpha = 0.95` EMA                       | `last = 0.95 * last + 0.05 * v`               |
| Slope via least-squares on smoothed Y    | Same closed-form `ssXy / ssXx`                |
| `relative_slope = slope / current_smooth`| identical                                      |
| `< -0.005 → improving / > 0.005 → div`   | identical                                      |
| `epochs_since_best > 50 → consider stop` | maps to `training_done`                       |
| `> 30 || abs(slope) < 0.001 → may stop`  | maps to `plateau`                             |
| `> 0.01 → diverging - check LR`          | maps to `diverging`                           |

The Status text the user sees in the email also matches exactly what
the chart-side `analyzeTraining()` (`src/components/spark/training-stats.tsx`)
shows them on the page. That's deliberate — same words, same wall.

If you tune the thresholds in either place, **also tune them in the other**.
Both are flagged with `// Mirror of training_monitor.py:649` comments so
future-us knows they're coupled.

---

## Files

```
src/lib/email.ts                                Resend transport + 3 templates
src/lib/training-alerts.ts                      Pure log → analysis function
src/app/api/cron/training-alerts/route.ts       The cron itself
src/app/api/spark/training-jobs/route.ts        Stashes prefs in env on submit
src/app/demo/jobs/new/page.tsx                  Email + 2 checkboxes (Step 4)
supabase/migrations/002_job_alerts.sql          Dedup table
vercel.json                                     Cron registration
```

`src/lib/training-alerts.ts` is intentionally pure (no DB / email / I/O
beyond the log fetch). It returns a `recommend: AlertKind[]` and the
cron route owns the persistence. That makes the heuristic trivial to
unit-test if we ever want to.

---

## Required setup before this can ship

**1. DB migration.** Run `supabase/migrations/002_job_alerts.sql` in the
Supabase SQL editor (Dashboard → SQL → New query). Creates one table:

```sql
CREATE TABLE job_alerts (
  id       uuid PRIMARY KEY,
  job_id   text NOT NULL,        -- Spark job UUID as text
  kind     text NOT NULL,        -- 'plateau' | 'training_done' | 'diverging'
  email    text NOT NULL,
  fired_at timestamptz NOT NULL DEFAULT now(),
  meta     jsonb NOT NULL DEFAULT '{}'::jsonb
);
```

No FK to `jobs(id)` because Spark jobs aren't in our DB yet (Spark is
the source of truth in the v1 flow). The unique-ish constraint is
enforced in code via the 4h cooldown lookup, not the schema, so we
don't need to schema-migrate when we tune the cooldown.

**2. Env vars** in `tunet-web/.env.local` (and in Vercel project settings
when we push to prod):

```
RESEND_API_KEY=re_...
RESEND_FROM=alerts@cleanplatefx.com   # must be a Resend-verified sender
PUBLIC_APP_URL=https://tunet.example.com   # for "view this job" links
CRON_SECRET=<same secret already used by billing-tick>
```

**3. Resend domain verification.** Add the DNS records Resend gives you
for whatever sender domain you pick. Without verification their API
returns 403.

**4. Vercel cron.** Already in `vercel.json`. Vercel auto-injects
`Authorization: Bearer ${CRON_SECRET}`. No further config.

---

## Testing without waiting 10 minutes

```bash
# Dry-run the cron locally — won't actually send unless RESEND_API_KEY is set,
# but exercises the full pipeline (job listing, log fetch, parse, analyze,
# DB insert).
curl -H "Authorization: Bearer $CRON_SECRET" \
     http://localhost:3000/api/cron/training-alerts | jq
```

Response shape:

```json
{
  "ok": true,
  "scanned": 3,
  "fired": 1,
  "skipped_no_email": 1,
  "skipped_no_log": 0,
  "skipped_cooldown": 1,
  "fires": [
    { "jobId": "abc-123", "kind": "plateau", "ok": true }
  ]
}
```

If `RESEND_API_KEY` is unset, `fires` entries will have
`{ ok: false, skipped: true }` — useful to validate the heuristic is
firing on a real run before pointing it at a real inbox.

---

## Open questions for Walt

1. **Is there a cheaper way to read partial `training.log`?** Spark's
   ShareSync WebDAV doesn't appear to honor `Range` headers in our
   testing. Right now I re-pull the whole log every tick per active
   job. At 10 KB × 6 ticks/hr × N active jobs, fine — but at scale or
   for very long runs (multi-MB logs), a `Range: bytes=N-` would be
   nice. Do you support it / plan to?

2. **Could Spark surface the `last_agent_heartbeat_at` for failed jobs
   that never started?** Right now I skip those (no log to parse). Not
   a bug, just curious if there's a signal we should be reading there.

3. **Is the env-var stash for prefs (`TUNET_ALERT_*`) durable?** I'm
   relying on Spark echoing back submit-time `env` in `GET /jobs/:id`
   forever — if you ever start truncating env on terminal jobs, the
   cron will see "no email configured" on jobs we already alerted on
   (no real harm — just stops working). If you want to surface that
   as a guarantee in the API docs that'd be useful.

4. **Future: would Spark consider sending its own platform-level alert
   webhook?** "Job has been running for 3h with no progress" or
   similar. Useful for cases where the app process is alive but stuck
   (we'd see flat loss, but Spark can also see "no new log lines for
   30min" which is a different signal). Not blocking — current setup
   handles the common case.

---

## What's NOT here (intentional v1 cuts)

- **No in-app notification center.** Email only for now. The job detail
  page already shows the same Trend / Plateau / Status chips live, so a
  user who's looking at the page doesn't need a duplicate notification.
- **No SMS / Slack / webhook.** All workflows the user cares about are
  rooted in their inbox. We can add Slack later as a transport in
  `email.ts` (rename to `notifications.ts`) without changing the cron.
- **No "training started" / "training complete" emails.** The user just
  submitted, they know it started. "Complete" is captured by Spark's
  own job-status emails when those exist (Walt's roadmap?).
- **No alert quiet hours.** Plateau alerts can fire at 3am. We expose a
  "snooze for 4h" link in the email if this turns into a real problem.
- **No retroactive scan.** The cron only looks at currently-active jobs.
  If a job finished + plateaued before the cron fired we don't email
  about it post-hoc. Not worth the complexity.

---

## Anti-spam notes

The 4h cooldown per (job, kind) is the main lever. Without it, plateau
emails would arrive every 10 min from epoch 31 onwards. Within 4h:

- Eric submits a job at 09:00.
- 13:30 — `training_done` fires (best is 51 epochs old). Email sent.
- 13:40 — cron runs. `epochsSinceBest` is now 52, still qualifies. But
  the 4h cooldown matches. Skipped.
- 17:30 — cooldown expires. If still in `training_done` state and Eric
  hasn't stopped the job, it fires again. The thinking: he's burning $5+
  of GPU at this point, a second email is justified.

If 4h is wrong in either direction we can change `COOLDOWN_MS` in the
cron route and redeploy — no migration.

---

## What changed in the chart-side code

To support the analyzer running on the same parsed series as the chart
(without a duplicate SSE connection per component), I extracted the
parse loop into a shared hook: `src/components/spark/use-training-stream.ts`.
It maintains a singleton `EventSource` per `jobId` keyed by a refcount
of subscribers. Both `<TrainingChart>` and `<TrainingStats>` consume it.

This is unrelated to alerts but worth noting because:

- The cron does NOT use this hook — it does its own one-shot
  log-text fetch. The hook is browser-only.
- If you ever want a server-side live tail (e.g. for a "is this job
  making progress" health check), use the SSE proxy
  (`src/app/api/spark/jobs/[id]/logs`) — same API the hook uses.
