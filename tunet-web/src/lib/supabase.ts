import { createClient } from '@supabase/supabase-js'
import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

// ── Server client (uses service role — server only, never in client bundle) ──

export function createServiceClient() {
  return createClient(supabaseUrl, supabaseServiceKey, {
    auth: { autoRefreshToken: false, persistSession: false },
  })
}

// ── Server component client (respects RLS, uses anon key + cookie session) ──

export async function createServerComponentClient() {
  const cookieStore = await cookies()
  return createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll()
      },
      setAll(cookiesToSet: { name: string; value: string; options?: Record<string, unknown> }[]) {
        try {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options as Parameters<typeof cookieStore.set>[2])
          )
        } catch {
          // Server components can't set cookies — ignore
        }
      },
    },
  })
}

// ── Storage helpers ───────────────────────────────────────────────────────────

export const STORAGE_BUCKET = 'tunet-jobs'

export function jobStoragePath(userId: string, jobId: string, filename: string): string {
  return `${userId}/${jobId}/${filename}`
}

export async function getSignedDownloadUrl(path: string): Promise<string | null> {
  const svc = createServiceClient()
  const { data, error } = await svc.storage
    .from(STORAGE_BUCKET)
    .createSignedUrl(path, 3600) // 1 hour
  if (error || !data) return null
  return data.signedUrl
}

export async function getSignedUploadUrl(path: string): Promise<{ signedUrl: string; token: string } | null> {
  const svc = createServiceClient()
  const { data, error } = await svc.storage
    .from(STORAGE_BUCKET)
    .createSignedUploadUrl(path)
  if (error || !data) return null
  return { signedUrl: data.signedUrl, token: data.token }
}

// Get a 24-hour signed download URL for pod bootstrap (longer TTL needed)
export async function getBootstrapSignedUrl(path: string): Promise<string | null> {
  const svc = createServiceClient()
  const { data, error } = await svc.storage
    .from(STORAGE_BUCKET)
    .createSignedUrl(path, 86400) // 24 hours
  if (error || !data) return null
  return data.signedUrl
}

// Admin bucket for the tunet code bundle
export const ADMIN_BUCKET = 'tunet-admin'

export async function getCodeBundleUrl(): Promise<string | null> {
  const path = process.env.TUNET_CODE_BUNDLE_PATH || 'admin/tunet-latest.tar.gz'
  const svc = createServiceClient()
  const { data, error } = await svc.storage
    .from(ADMIN_BUCKET)
    .createSignedUrl(path, 86400)
  if (error || !data) return null
  return data.signedUrl
}
