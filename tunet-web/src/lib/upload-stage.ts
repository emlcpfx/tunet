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
 * On a 200-500 MB .pth this can take 10-60 s depending on the network. The
 * onProgress callback is called once at the end (FormData doesn't expose
 * progress events on fetch); use a spinner / "uploading…" placeholder in
 * the caller instead of trying to drive a progress bar.
 */
export async function uploadCheckpoint(
  file: File,
): Promise<{ stageId: string; filename: string; bytes: number }> {
  const fd = new FormData()
  fd.set('role', 'checkpoint')
  fd.append('files', file, file.name)

  const res = await fetch('/api/spark/upload-stage', { method: 'POST', body: fd })
  if (!res.ok) {
    const j = await res.json().catch(() => ({}))
    throw new Error(j.error ?? `upload-stage HTTP ${res.status}`)
  }
  const json = await res.json() as {
    stageId: string
    received: { role: string; files: number; bytes: number }
  }
  return {
    stageId:  json.stageId,
    filename: file.name,
    bytes:    json.received?.bytes ?? file.size,
  }
}
