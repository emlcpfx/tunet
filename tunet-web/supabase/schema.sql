-- ============================================================
-- TuNet Cloud — Supabase / Postgres Schema
-- Run this in the Supabase SQL editor (Dashboard → SQL → New query)
-- ============================================================

-- ── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Tables ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
  id                    text        PRIMARY KEY,   -- Clerk user ID
  email                 text        NOT NULL,
  name                  text,
  credit_balance_cents  integer     NOT NULL DEFAULT 0,
  is_admin              boolean     NOT NULL DEFAULT false,
  created_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS gpu_pricing (
  gpu_type_id           text        PRIMARY KEY,   -- RunPod gpuTypeId
  display_name          text        NOT NULL,
  short_key             text        NOT NULL,
  vram_gb               integer,
  platform_cost_per_hr  numeric(10,4) NOT NULL,
  runpod_cost_per_hr    numeric(10,4) NOT NULL,
  is_available          boolean     NOT NULL DEFAULT true,
  tier                  text        NOT NULL DEFAULT 'standard',
  sort_order            integer     NOT NULL DEFAULT 99
);

CREATE TABLE IF NOT EXISTS jobs (
  id                    uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               text        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name                  text        NOT NULL,
  pod_id                text,
  status                text        NOT NULL DEFAULT 'pending',
  gpu_type_id           text        NOT NULL,
  gpu_display_name      text,
  runpod_cost_per_hr    numeric(10,4),
  platform_cost_per_hr  numeric(10,4),
  accumulated_cost_cents integer     NOT NULL DEFAULT 0,
  billing_last_tick_at  timestamptz,
  config_path           text,
  src_zip_path          text,
  dst_zip_path          text,
  checkpoint_path       text,
  container_disk_gb     integer     NOT NULL DEFAULT 50,
  volume_gb             integer     NOT NULL DEFAULT 100,
  started_at            timestamptz,
  ended_at              timestamptz,
  created_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS billing_events (
  id                      uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                 text        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  job_id                  uuid        REFERENCES jobs(id) ON DELETE SET NULL,
  type                    text        NOT NULL,   -- top_up|compute_charge|manual_adjustment|refund
  amount_cents            integer     NOT NULL,   -- positive=credit, negative=debit
  description             text,
  stripe_payment_intent   text,
  created_at              timestamptz NOT NULL DEFAULT now()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_jobs_user_id        ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status         ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_pod_id         ON jobs(pod_id) WHERE pod_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_billing_user_id     ON billing_events(user_id);
CREATE INDEX IF NOT EXISTS idx_billing_job_id      ON billing_events(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_billing_stripe_pi   ON billing_events(stripe_payment_intent) WHERE stripe_payment_intent IS NOT NULL;

-- ── Row-Level Security ────────────────────────────────────────────────────────
ALTER TABLE users         ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs          ENABLE ROW LEVEL SECURITY;
ALTER TABLE billing_events ENABLE ROW LEVEL SECURITY;

-- Service role bypasses RLS — used by all server-side operations
-- Users can only see their own data (anon/authenticated role)

CREATE POLICY "Users can read own record"
  ON users FOR SELECT
  USING (auth.uid()::text = id);

CREATE POLICY "Users can read own jobs"
  ON jobs FOR SELECT
  USING (auth.uid()::text = user_id);

CREATE POLICY "Users can read own billing events"
  ON billing_events FOR SELECT
  USING (auth.uid()::text = user_id);

-- gpu_pricing is public readable
ALTER TABLE gpu_pricing ENABLE ROW LEVEL SECURITY;
CREATE POLICY "GPU pricing is public"
  ON gpu_pricing FOR SELECT
  USING (true);

-- ── RPC Functions ─────────────────────────────────────────────────────────────

-- Add credits atomically (used by Stripe webhook)
CREATE OR REPLACE FUNCTION add_credits(p_user_id text, p_amount_cents integer)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  UPDATE users
  SET    credit_balance_cents = credit_balance_cents + p_amount_cents
  WHERE  id = p_user_id;
END;
$$;

-- Adjust credits (admin use — can be negative for manual corrections)
CREATE OR REPLACE FUNCTION adjust_credits(
  p_user_id     text,
  p_amount_cents integer,
  p_description  text DEFAULT 'Manual adjustment'
)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  UPDATE users
  SET    credit_balance_cents = GREATEST(0, credit_balance_cents + p_amount_cents)
  WHERE  id = p_user_id;

  INSERT INTO billing_events (user_id, type, amount_cents, description)
  VALUES (p_user_id, 'manual_adjustment', p_amount_cents, p_description);
END;
$$;

-- ── Seed GPU Pricing ──────────────────────────────────────────────────────────
INSERT INTO gpu_pricing
  (gpu_type_id, display_name, short_key, vram_gb, runpod_cost_per_hr, platform_cost_per_hr, is_available, tier, sort_order)
VALUES
  ('NVIDIA GeForce RTX 4090',                    'RTX 4090',              '4090',       24, 0.74, 1.09, true, 'standard',    10),
  ('NVIDIA A40',                                 'A40',                   'a40',        48, 0.79, 1.19, true, 'standard',    20),
  ('NVIDIA L40S',                                'L40S',                  'l40s',       48, 1.14, 1.79, true, 'recommended', 30),
  ('NVIDIA RTX PRO 6000 Blackwell Server Edition','RTX PRO 6000 Blackwell','rtxpro6000', 96, 1.99, 2.99, true, 'premium',     40),
  ('NVIDIA A100 80GB PCIe',                      'A100 80GB',             'a100',       80, 2.99, 4.50, true, 'pro',         50),
  ('NVIDIA A100-SXM4-80GB',                      'A100 SXM4 80GB',        'a100sxm',    80, 3.49, 5.25, true, 'pro',         60)
ON CONFLICT (gpu_type_id) DO UPDATE
  SET display_name         = EXCLUDED.display_name,
      platform_cost_per_hr = EXCLUDED.platform_cost_per_hr,
      runpod_cost_per_hr   = EXCLUDED.runpod_cost_per_hr,
      is_available         = EXCLUDED.is_available,
      tier                 = EXCLUDED.tier,
      sort_order           = EXCLUDED.sort_order;

-- ── Storage Buckets (run in Storage dashboard or via API) ─────────────────────
-- These cannot be created via SQL — run in Supabase dashboard:
--
-- 1. Create bucket "tunet-jobs"  — private, max file size 10 GB
-- 2. Create bucket "tunet-admin" — private, max file size 2 GB
--
-- Storage RLS Policies for "tunet-jobs":
-- INSERT: auth.uid()::text = (storage.foldername(name))[1]  (users upload to own folder)
-- SELECT: auth.uid()::text = (storage.foldername(name))[1]  (users download own files)
-- DELETE: auth.uid()::text = (storage.foldername(name))[1]  (users delete own files)
--
-- "tunet-admin" bucket: service role only (no public policies)
