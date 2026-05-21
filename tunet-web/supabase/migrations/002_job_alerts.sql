-- ============================================================
-- Spark Flint — Job alerts (training plateau / done notifications)
-- Apply via Supabase SQL editor.
-- ============================================================

-- Per-(job, kind) dedup record. The cron scanner inserts a row when it fires
-- an alert for a job; the unique constraint blocks repeats of the same kind
-- within the suppression window we enforce in code.
CREATE TABLE IF NOT EXISTS job_alerts (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id      text        NOT NULL,           -- Spark job id (uuid as text — no jobs FK; jobs may not be in our DB)
  kind        text        NOT NULL,           -- 'plateau' | 'training_done' | 'diverging'
  email       text        NOT NULL,
  fired_at    timestamptz NOT NULL DEFAULT now(),
  -- Snapshot of the metric values at fire-time, for debugging / future UIs
  meta        jsonb       NOT NULL DEFAULT '{}'::jsonb
);

-- Used by the cron poller to avoid double-firing. We also enforce a
-- soft cooldown window in code (4h) so plateau alerts don't repeat at
-- 31, 32, 33 epochs.
CREATE INDEX IF NOT EXISTS idx_job_alerts_job_kind  ON job_alerts (job_id, kind, fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_alerts_fired_at  ON job_alerts (fired_at DESC);
