/**
 * Client-side helper that batches a FolderPickerResult into multiple
 * /api/spark/upload-stage POSTs and returns the resulting stageId.
 *
 * Extracted from new-job/page.tsx so multiple flows (new-job, benchmark)
 * can share one upload codepath. Behaviour is identical to the original:
 * files are grouped per role, batched up to MAX_BATCH_BYTES, and sent
 * sequentially so the server's tmpdir doesn't get hit with parallel
 * 200 MB writes.
 */

import type { FolderPickerResult } from '@/components/spark/folder-picker'

// Per-POST cap. Next.js buffers the whole multipart request body in RAM before
// the route handler runs, so this is effectively the peak heap per batch — keep
// it well under the prod box's ~1 GB (a 200 MB batch + a concurrent tenant spike
// is what OOM-killed the .pth upload). 64 MB → flat memory; more POSTs is the
// cheap price. (A single source frame larger than this still goes in its own
// batch.) Long-term fix is direct browser→ShareSync upload, bypassing the box.
export const MAX_BATCH_BYTES = 64 * 1024 * 1024

export async function uploadStage(
  picked: FolderPickerResult,
  onProgress: (sent: number, total: number, batchIdx: number, totalBatches: number) => void,
): Promise<string> {
  const batches: { role: string; files: File[]; bytes: number }[] = []

  function addRole(role: string, entries: { name: string; size: number; file: File }[]) {
    if (entries.length === 0) return
    let cur: File[] = []
    let curBytes = 0
    for (const e of entries) {
      if (curBytes + e.size > MAX_BATCH_BYTES && cur.length > 0) {
        batches.push({ role, files: cur, bytes: curBytes })
        cur = []
        curBytes = 0
      }
      cur.push(e.file)
      curBytes += e.size
    }
    if (cur.length > 0) batches.push({ role, files: cur, bytes: curBytes })
  }

  addRole('src',     picked.src)
  addRole('dst',     picked.dst)
  addRole('val_src', picked.valSrc)
  addRole('val_dst', picked.valDst)
  addRole('mask',    picked.mask)

  const total = batches.reduce((s, b) => s + b.bytes, 0)
  let sent = 0
  let stageId: string | null = null

  for (let i = 0; i < batches.length; i++) {
    const b = batches[i]
    const fd = new FormData()
    fd.set('role', b.role)
    if (stageId) fd.set('stageId', stageId)
    for (const f of b.files) fd.append('files', f, f.name)

    const res = await fetch('/api/spark/upload-stage', { method: 'POST', body: fd })
    if (!res.ok) {
      const j = await res.json().catch(() => ({}))
      throw new Error(j.error ?? `upload-stage HTTP ${res.status}`)
    }
    const json = await res.json() as { stageId: string }
    stageId = json.stageId
    sent += b.bytes
    onProgress(sent, total, i + 1, batches.length)
  }

  if (!stageId) throw new Error('No batches uploaded — picker had no files')
  return stageId
}

/**
 * Upload a single .pth checkpoint file for local-resume. Returns the stageId
 * the training-jobs route can read it from. Separate from uploadStage()
 * because there's no folder picker / role decomposition — just one file with
 * a known role and basename.
 *
 * Sent as RAW octet-stream chunks (not multipart) appended server-side into one
 * file. We slice the .pth into CHECKPOINT_CHUNK_BYTES pieces and POST them
 * sequentially because Next.js buffers each request body fully in RAM before the
 * route handler runs — a single ~1 GB body OOM-kills the small (1 GB) prod VPS.
 * Chunking caps the server's buffered body (and our slice) at one chunk. The
 * first chunk mints the stage; the rest append to it. Metadata rides in headers.
 *
 * onProgress (optional) reports bytes sent so the caller can show a bar; without
 * it, show a "uploading…" spinner. A 200 MB–1 GB .pth takes ~10-90 s on network.
 */
export const CHECKPOINT_CHUNK_BYTES = 32 * 1024 * 1024

export async function uploadCheckpoint(
  file: File,
  onProgress?: (sent: number, total: number) => void,
): Promise<{ stageId: string; filename: string; bytes: number }> {
  return uploadRawChunked(file, 'checkpoint', onProgress)
}

/**
 * EZ-Comfy input — same chunked raw upload, staged under 'comfy_input' by
 * default. For two-input presets (face-swap, VACE mask), pass
 * `{ role: 'comfy_input2', stageId }` to drop the SECONDARY file into the SAME
 * stage as the primary clip, so submit can resolve both from one stageId.
 */
export async function uploadComfyInput(
  file: File,
  onProgress?: (sent: number, total: number) => void,
  opts?: { role?: string; stageId?: string },
): Promise<{ stageId: string; filename: string; bytes: number }> {
  return uploadRawChunked(file, opts?.role ?? 'comfy_input', onProgress, opts?.stageId)
}

/**
 * EZ-Comfy BATCH image-sequence folder — many frames staged under
 * comfy_seq/<subdir>/ in one stage (one subdir per folder, so several sequences
 * don't collide by basename). Frames go up as 64 MB multipart batches (same
 * memory discipline as the training folder uploader). Returns the stageId so the
 * caller chains the next folder / clip into the same stage.
 */
export async function uploadComfySequence(
  files: File[],
  subdir: string,
  stageId: string | undefined,
  onProgress?: (sent: number, total: number) => void,
): Promise<{ stageId: string; count: number; bytes: number }> {
  let sid = stageId
  const total = files.reduce((s, f) => s + f.size, 0)
  let sent = 0
  let cur: File[] = []
  let curBytes = 0

  const flush = async () => {
    if (cur.length === 0) return
    const fd = new FormData()
    fd.set('role', 'comfy_seq')
    fd.set('subdir', subdir)
    if (sid) fd.set('stageId', sid)
    for (const f of cur) fd.append('files', f, f.name)
    const res = await fetch('/api/spark/upload-stage', { method: 'POST', body: fd })
    if (!res.ok) {
      const j = await res.json().catch(() => ({}))
      throw new Error(j.error ?? `upload-stage HTTP ${res.status}`)
    }
    const json = await res.json() as { stageId: string }
    sid = json.stageId
    sent += curBytes
    onProgress?.(sent, total)
    cur = []
    curBytes = 0
  }

  for (const f of files) {
    if (curBytes + f.size > MAX_BATCH_BYTES && cur.length > 0) await flush()
    cur.push(f)
    curBytes += f.size
  }
  await flush()
  if (!sid) throw new Error('no frames uploaded')
  return { stageId: sid, count: files.length, bytes: total }
}

async function uploadRawChunked(
  file: File,
  role: string,
  onProgress?: (sent: number, total: number) => void,
  initialStageId?: string,
): Promise<{ stageId: string; filename: string; bytes: number }> {
  // An initial stageId appends this file to an existing stage (under a new role
  // dir); the first chunk still uses 'create' mode so it writes a fresh file.
  let stageId = initialStageId ?? ''
  const total = file.size
  for (let start = 0; start < total; start += CHECKPOINT_CHUNK_BYTES) {
    const end = Math.min(start + CHECKPOINT_CHUNK_BYTES, total)
    const headers: Record<string, string> = {
      'content-type': 'application/octet-stream',
      'x-role':       role,
      'x-filename':   encodeURIComponent(file.name),
      'x-chunk-mode': start === 0 ? 'create' : 'append',
    }
    if (stageId) headers['x-stage-id'] = stageId

    const res = await fetch('/api/spark/upload-stage', {
      method:  'POST',
      headers,
      body: file.slice(start, end),
    })
    if (!res.ok) {
      const j = await res.json().catch(() => ({}))
      throw new Error(j.error ?? `upload-stage HTTP ${res.status}`)
    }
    const json = await res.json() as { stageId: string }
    stageId = json.stageId
    onProgress?.(end, total)
  }
  if (!stageId) throw new Error('file is empty')
  return { stageId, filename: file.name, bytes: total }
}
