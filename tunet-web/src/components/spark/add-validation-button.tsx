'use client'
import { useState } from 'react'
import { FolderPicker, type FolderPickerResult } from './folder-picker'

/**
 * Add validation data to a RUNNING training job. The user picks a project folder;
 * we take its `val_src`/`val_dst` subfolders (falling back to `src`/`dst` if the
 * folder is just a plain pair set), upload the files to the job's ShareSync
 * control area in batches, then commit a manifest the trainer polls. The job
 * ADDS them to its current validation set and rebuilds the val dataloader at its
 * next checkpoint — no restart. Only rendered for live training jobs.
 */
const BATCH = 25 // files per upload request — val sets are small; keeps each POST modest

type FileEntry = { name: string; file: File }

export function AddValidationButton({ jobId }: { jobId: string }) {
  const [open, setOpen] = useState(false)
  const [picked, setPicked] = useState<FolderPickerResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<string | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // Prefer val_src/val_dst; fall back to a plain src/dst pair set picked as validation.
  const valSrc: FileEntry[] = picked ? (picked.valSrc.length ? picked.valSrc : picked.src) : []
  const valDst: FileEntry[] = picked ? (picked.valDst.length ? picked.valDst : picked.dst) : []

  async function uploadRole(batchId: string, role: 'val_src' | 'val_dst', entries: FileEntry[]) {
    for (let i = 0; i < entries.length; i += BATCH) {
      const slice = entries.slice(i, i + BATCH)
      const fd = new FormData()
      fd.append('batchId', batchId)
      fd.append('role', role)
      for (const e of slice) fd.append('files', e.file, e.name)
      setProgress(`Uploading ${role} ${Math.min(i + BATCH, entries.length)}/${entries.length}…`)
      const res = await fetch(`/api/spark/jobs/${jobId}/add-validation`, { method: 'POST', body: fd })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
    }
  }

  async function apply() {
    if (valSrc.length === 0) { setErr('Pick a folder with a val_src (or src) subfolder of images'); return }
    setBusy(true); setErr(null); setMsg(null)
    const batchId = `b${Date.now()}`
    try {
      await uploadRole(batchId, 'val_src', valSrc)
      if (valDst.length > 0) await uploadRole(batchId, 'val_dst', valDst)
      setProgress('Committing…')
      const res = await fetch(`/api/spark/jobs/${jobId}/add-validation?commit=1`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batchId, src: valSrc.map(e => e.name), dst: valDst.map(e => e.name) }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`)
      setMsg(data.note ?? 'Validation data added.')
      setOpen(false); setPicked(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'failed')
    } finally {
      setBusy(false); setProgress(null)
    }
  }

  // Collapsed: a header-friendly button. The expanded picker is a centered modal
  // (the folder picker is too tall to live inline in the header action row).
  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={() => { setOpen(true); setMsg(null) }}
        className="px-3 py-1.5 rounded-md text-xs font-semibold border border-[#e5e7eb] text-[#374151] hover:bg-[#F9FAFB] transition-colors"
        title="Upload validation frames to this running job — added to its validation set without a restart"
      >
        Add validation data
      </button>
      {msg && <p className="text-xs text-[#16a34a] max-w-[260px] text-right">{msg}</p>}
      {open && <Modal>{panel()}</Modal>}
    </div>
  )

  function panel() {
    return (
    <div className="w-full max-w-md bg-white border border-[#e5e7eb] rounded-lg p-4 space-y-3 shadow-xl">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-[#374151]">Add validation data</p>
        <button
          onClick={() => { setOpen(false); setErr(null); setPicked(null) }}
          className="text-xs text-[#6b7280] hover:text-[#374151]"
          disabled={busy}
        >
          ✕
        </button>
      </div>
      <p className="text-xs text-[#6b7280]">
        Pick a folder with <code className="font-mono">val_src</code>/<code className="font-mono">val_dst</code> subfolders
        (or a plain <code className="font-mono">src</code>/<code className="font-mono">dst</code> pair set). New frames are
        <strong> added</strong> to the job&apos;s current validation set — it rebuilds at the next checkpoint, no restart.
        <code className="font-mono">val_dst</code> is optional (without it you get validation previews but no loss metrics).
      </p>

      <FolderPicker onPicked={setPicked} />

      {picked && (
        <p className="text-xs text-[#374151]">
          Ready to add: <strong>{valSrc.length}</strong> val_src
          {valDst.length > 0 ? <> · <strong>{valDst.length}</strong> val_dst</> : <> (preview-only — no val_dst)</>}
        </p>
      )}

      {progress && <p className="text-xs text-[#7E3AF2]">{progress}</p>}
      {err && <p className="text-xs text-[#EF4444]">{err}</p>}

      <div className="flex justify-end gap-2">
        <button
          onClick={() => { setOpen(false); setErr(null); setPicked(null) }}
          disabled={busy}
          className="px-3 py-1.5 rounded-md text-xs text-[#6b7280] hover:bg-[#F9FAFB] disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          onClick={() => void apply()}
          disabled={busy || valSrc.length === 0}
          className="px-3 py-1.5 rounded-md text-xs font-semibold bg-[#7E3AF2] text-white hover:bg-[#6D28D9] disabled:opacity-50"
        >
          {busy ? 'Adding…' : 'Add to job'}
        </button>
      </div>
    </div>
    )
  }
}

/** Minimal centered modal — backdrop + card, no portal (the parent is mounted
 *  in the page body, so a fixed overlay renders fine). */
function Modal({ children }: { children: React.ReactNode }) {
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 p-4 overflow-y-auto">
      <div className="mt-16 text-left">{children}</div>
    </div>
  )
}
