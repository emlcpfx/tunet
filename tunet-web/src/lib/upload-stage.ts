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

export const MAX_BATCH_BYTES = 200 * 1024 * 1024

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
  let stageId = ''
  const total = file.size
  for (let start = 0; start < total; start += CHECKPOINT_CHUNK_BYTES) {
    const end = Math.min(start + CHECKPOINT_CHUNK_BYTES, total)
    const headers: Record<string, string> = {
      'content-type': 'application/octet-stream',
      'x-role':       'checkpoint',
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
  if (!stageId) throw new Error('checkpoint file is empty')
  return { stageId, filename: file.name, bytes: total }
}
