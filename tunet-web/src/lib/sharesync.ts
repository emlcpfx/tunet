/**
 * Minimal ShareSync (WebDAV) helpers for the EZ-Comfy / live-export control
 * channel. Writes small JSON control files under a dedicated `_tunet_control/`
 * path that the Spark agent never touches (it only syncs each job's /output),
 * so they persist for the running job to poll.
 *
 * The bearer stays server-side (same token spark.ts uses on api.* and files.*).
 */
import 'server-only'
import { getToken, encodePathSegments } from './spark'

export function spaceBase(): string {
  return (process.env.SPARK_FILES_BASE_URL ?? '').replace(/\/+$/, '')
}

async function dav(method: string, url: string, body?: BodyInit): Promise<Response> {
  const tok = await getToken()
  return fetch(url, {
    method,
    headers: {
      'Authorization': `Bearer ${tok}`,
      ...(body !== undefined ? { 'Content-Type': 'application/octet-stream' } : {}),
    },
    ...(body !== undefined ? { body } : {}),
    cache: 'no-store',
  })
}

async function mkcol(url: string): Promise<void> {
  try { await dav('MKCOL', url) } catch { /* already exists / racing — fine */ }
}

/** PUT a JSON control file at `_tunet_control/<key>/<file>`, creating the dirs. */
export async function putControlFile(key: string, file: string, obj: unknown): Promise<void> {
  const base = spaceBase()
  if (!base) throw new Error('SPARK_FILES_BASE_URL not configured')
  const k = encodeURIComponent(key)
  await mkcol(`${base}/_tunet_control`)
  await mkcol(`${base}/_tunet_control/${k}`)
  const res = await dav('PUT', `${base}/_tunet_control/${k}/${file}`, JSON.stringify(obj))
  if (!res.ok) {
    const t = await res.text().catch(() => '')
    throw new Error(`ShareSync PUT ${file} → HTTP ${res.status} ${t.slice(0, 120)}`)
  }
}

/**
 * PUT raw bytes at `_tunet_control/<key>/<seg>/<seg>/…`, MKCOL-ing every parent
 * collection along the way. Used to deliver post-submit validation files into a
 * running job's control area (the trainer downloads them from there). All but
 * the last segment are treated as directories.
 */
export async function putControlBytes(key: string, segs: string[], body: ArrayBuffer): Promise<void> {
  const base = spaceBase()
  if (!base) throw new Error('SPARK_FILES_BASE_URL not configured')
  if (segs.length === 0) throw new Error('putControlBytes: empty path')
  let cur = `${base}/_tunet_control`
  await mkcol(cur)
  cur = `${cur}/${encodeURIComponent(key)}`
  await mkcol(cur)
  for (const seg of segs.slice(0, -1)) {
    cur = `${cur}/${encodeURIComponent(seg)}`
    await mkcol(cur)
  }
  const url = `${cur}/${encodeURIComponent(segs[segs.length - 1])}`
  const res = await dav('PUT', url, body)
  if (!res.ok) {
    const t = await res.text().catch(() => '')
    throw new Error(`ShareSync PUT ${segs.join('/')} → HTTP ${res.status} ${t.slice(0, 120)}`)
  }
}

/**
 * The job's output WebDAV base = space base + its output_share_sync_path.
 * The path segments are percent-encoded (e.g. "Spark Fuse Jobs" → "Spark%20Fuse%20Jobs")
 * — without this the raw space lands in the job's meta.json and the in-container
 * urllib upload rejects it ("URL can't contain control characters"). spaceBase()
 * is already encoded (%24 for the space-id $); encodePathSegments is idempotent
 * so it won't double-encode it.
 */
export function outputDavUrl(outputSharePath: string | null | undefined): string | null {
  if (!outputSharePath) return null
  const rel = outputSharePath.replace(/^\/+/, '').replace(/\/+$/, '')
  return `${spaceBase()}/${encodePathSegments(rel)}`
}
