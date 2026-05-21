'use client'

/**
 * Folder picker for training data — uses <input webkitdirectory>.
 *
 * Behavior: user picks a project folder; we scan its top-level subdirectories
 * for any folder whose name matches one of the canonical aliases (mirrors
 * gui/data_tab.py to stay compatible with existing TuNet projects):
 *
 *     src     — src, source, in, input
 *     dst     — dst, dest, destination, out, output, target
 *     val_src — val_src, val_source, val_input
 *     val_dst — val_dst, val_dest, val_target
 *     mask    — mask, masks, matte, mattes
 *     model/  — checkpoints output (just a hint, not used here)
 *
 * For each detected folder we count image files and dimensions of the first
 * sample. We then compute matched src↔dst pairs by filename basename.
 *
 * The picker doesn't upload — it just discovers structure and reports back.
 * Actual data upload happens via ShareSync (next step).
 */

import { useCallback, useRef, useState } from 'react'

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff', '.bmp', '.webp'])

// Aliases mirror gui/data_tab.py — see _SRC_NAMES, _DST_NAMES, etc.
const ROLE_ALIASES = {
  src:     ['src', 'source', 'in', 'input'],
  dst:     ['dst', 'dest', 'destination', 'out', 'output', 'target'],
  val_src: ['val_src', 'val_source', 'val_input'],
  val_dst: ['val_dst', 'val_dest', 'val_target'],
  mask:    ['mask', 'masks', 'matte', 'mattes'],
} as const

type Role = keyof typeof ROLE_ALIASES

/** Map a literal subfolder name (lowercased) to its canonical role, or null */
function classifyFolder(name: string): Role | null {
  const lc = name.toLowerCase()
  for (const [role, aliases] of Object.entries(ROLE_ALIASES) as [Role, readonly string[]][]) {
    if (aliases.includes(lc)) return role
  }
  return null
}

interface FolderInfo {
  /** First top-level segment of webkitRelativePath — the picked folder name */
  rootName:    string
  /** Total bytes across all image files in classified folders */
  totalBytes:  number
  /** Per-role detection: actual folder name used + file count */
  roles:       Partial<Record<Role, { folderName: string; entries: FileEntry[] }>>
  /** Number of src↔dst pairs that share a filename basename */
  pairCount:   number
  /** Image dimensions of first src sample (best-effort; null if can't decode) */
  sampleDims:  { w: number; h: number } | null
  /** First src sample's extension, e.g. '.exr' */
  sampleExt:   string | null
  /** src filenames with no dst sharing their basename (the unpaired ones) */
  unmatchedSrc: string[]
}

interface FileEntry {
  name: string
  size: number
  file: File
}

export interface FolderPickerResult {
  rootName:   string
  /** Actual folder name detected for each role (e.g. 'input' or 'src') */
  folderNames: Partial<Record<Role, string>>
  src:        FileEntry[]
  dst:        FileEntry[]
  valSrc:     FileEntry[]
  valDst:     FileEntry[]
  mask:       FileEntry[]
  pairCount:  number
  totalBytes: number
  sampleDims: { w: number; h: number } | null
  sampleExt:  string | null
}

interface FolderPickerProps {
  onPicked: (result: FolderPickerResult) => void
}

export function FolderPicker({ onPicked }: FolderPickerProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [info, setInfo] = useState<FolderInfo | null>(null)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // webkitdirectory / directory / mozdirectory are non-standard so React's
  // type system doesn't expose them as JSX props. They MUST be set on every
  // mount of the underlying <input>, including post-HMR re-renders. The
  // earlier useEffect(() => {...}, []) pattern was fragile: if React swapped
  // the DOM node out for any reason (sibling re-render, suspense boundary,
  // strict-mode remount), the new node would lack the attributes and the
  // picker would silently degrade to a file picker. A ref callback runs
  // synchronously on every node-attach, so the attributes can't be lost.
  const setInputRef = useCallback((el: HTMLInputElement | null) => {
    inputRef.current = el
    if (!el) return
    el.setAttribute('webkitdirectory', '')
    el.setAttribute('directory', '')
    el.setAttribute('mozdirectory', '')
  }, [])

  async function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files
    if (!files || files.length === 0) return

    setScanning(true)
    setError(null)

    try {
      // Group every classified subfolder under its role.
      // We track the *actual* folder name used (e.g. 'input') so we can
      // surface it back to the user, and so the ShareSync path suggestion
      // matches what they'll need to upload.
      const roleEntries:    Partial<Record<Role, FileEntry[]>>  = {}
      const roleFolderName: Partial<Record<Role, string>>       = {}
      let rootName = ''
      let totalBytes = 0

      for (let i = 0; i < files.length; i++) {
        const f = files[i]
        // webkitRelativePath: "rootName/subdir/.../filename"
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath ?? ''
        const parts = rel.split('/').filter(Boolean)
        if (parts.length < 2) continue   // skip files at root or weird depth

        if (!rootName) rootName = parts[0]
        const subdir = parts[1]

        const role = classifyFolder(subdir)
        if (!role) continue   // skip unrecognized folders (model/, configs/, etc.)

        const ext = extractExt(f.name)
        if (!IMAGE_EXTS.has(ext)) continue

        if (!roleEntries[role])    roleEntries[role]    = []
        if (!roleFolderName[role]) roleFolderName[role] = subdir
        roleEntries[role]!.push({ name: f.name, size: f.size, file: f })
        totalBytes += f.size
      }

      const src    = roleEntries.src     ?? []
      const dst    = roleEntries.dst     ?? []
      const valSrc = roleEntries.val_src ?? []
      const valDst = roleEntries.val_dst ?? []
      const mask   = roleEntries.mask    ?? []

      // Compute matched pairs (filename basename match). Keep the names of the
      // src files that DON'T have a dst so the UI can name them, not just count.
      const dstSet = new Set(dst.map(e => stripExt(e.name).toLowerCase()))
      const pairCount = src.filter(e => dstSet.has(stripExt(e.name).toLowerCase())).length
      const unmatchedSrc = src
        .filter(e => !dstSet.has(stripExt(e.name).toLowerCase()))
        .map(e => e.name)
        .sort((a, b) => a.localeCompare(b))

      // Sample dimensions from first src image (if possible)
      let sampleDims: { w: number; h: number } | null = null
      let sampleExt: string | null = null
      if (src.length > 0) {
        sampleExt = extractExt(src[0].name)
        // Only decode common formats — EXR/TIF can't be decoded by browser
        if (['.png', '.jpg', '.jpeg', '.webp', '.bmp'].includes(sampleExt)) {
          sampleDims = await decodeDimensions(src[0].file)
        }
      }

      // Build role map for the info panel
      const rolesInfo: FolderInfo['roles'] = {}
      for (const role of Object.keys(roleEntries) as Role[]) {
        rolesInfo[role] = {
          folderName: roleFolderName[role]!,
          entries:    roleEntries[role]!,
        }
      }

      const result: FolderPickerResult = {
        rootName,
        folderNames: roleFolderName,
        src, dst, valSrc, valDst, mask,
        pairCount, totalBytes, sampleDims, sampleExt,
      }

      setInfo({ rootName, totalBytes, roles: rolesInfo, pairCount, sampleDims, sampleExt, unmatchedSrc })
      onPicked(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to scan folder')
    } finally {
      setScanning(false)
    }
  }

  function reset() {
    setInfo(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="space-y-3">
      {!info ? (
        <label className="block">
          <input
            ref={setInputRef}
            type="file"
            multiple
            onChange={onChange}
            className="hidden"
            disabled={scanning}
          />
          <div className="border-2 border-dashed border-[#D1D5DB] rounded-lg p-6 text-center hover:border-[#ae69f4] hover:bg-[#F7F4FC] transition-colors cursor-pointer">
            <div className="flex flex-col items-center gap-2">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-[#9ca3af]">
                <path d="M4 4h6l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z" />
              </svg>
              <p className="text-sm text-[#374151]">
                {scanning ? 'Scanning…' : <><strong>Click to pick a project folder</strong></>}
              </p>
              <p className="text-xs text-[#9ca3af]">
                Auto-detects subfolders named <code className="bg-white px-1 rounded font-mono">src</code>/<code className="bg-white px-1 rounded font-mono">input</code>/<code className="bg-white px-1 rounded font-mono">source</code> and <code className="bg-white px-1 rounded font-mono">dst</code>/<code className="bg-white px-1 rounded font-mono">output</code>/<code className="bg-white px-1 rounded font-mono">target</code>. Optional: <code className="bg-white px-1 rounded font-mono">val_src</code>, <code className="bg-white px-1 rounded font-mono">val_dst</code>, <code className="bg-white px-1 rounded font-mono">mask</code>.
              </p>
              <p className="text-[11px] text-[#9ca3af] italic mt-1">
                Your browser will ask once to confirm the upload — that&apos;s normal for folder uploads. We only read images from the canonical subfolders above; nothing else leaves your machine.
              </p>
            </div>
          </div>
        </label>
      ) : (
        <div className="bg-[#F7F4FC] border border-[#e9d5ff] rounded-lg p-4">
          <div className="flex items-start justify-between mb-3">
            <div className="min-w-0">
              <p className="text-xs font-semibold text-[#7E3AF2] uppercase tracking-wider">Picked</p>
              <p className="text-sm font-mono text-[#111827] truncate mt-0.5">{info.rootName}/</p>
            </div>
            <button
              type="button"
              onClick={reset}
              className="text-xs text-[#7E3AF2] hover:text-[#6C2BD9] flex-shrink-0 ml-2"
            >
              Pick a different folder
            </button>
          </div>

          {/* Detected subdirs — show actual folder name found, with role badge */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {(['src', 'dst', 'val_src', 'val_dst'] as const).map(role => {
              const r = info.roles[role]
              const present = !!r && r.entries.length > 0
              const folder = r?.folderName ?? ''
              return (
                <div
                  key={role}
                  className={`px-3 py-2 rounded border ${
                    present
                      ? 'bg-white border-[#e5e7eb] text-[#111827]'
                      : 'bg-[#F9FAFB] border-dashed border-[#D1D5DB] text-[#9ca3af]'
                  }`}
                >
                  <p className="text-[10px] uppercase tracking-wider font-semibold opacity-70">
                    {role.replace('_', ' ')}
                  </p>
                  <p className="text-xs font-mono mt-0.5 truncate">
                    {present ? `${folder}/  · ${r.entries.length}` : 'not found'}
                  </p>
                </div>
              )
            })}
          </div>
          {info.roles.mask && (
            <p className="mt-2 text-xs text-[#16A34A]">
              ✓ also detected mask folder: <code className="bg-white px-1 rounded font-mono">{info.roles.mask.folderName}/</code> ({info.roles.mask.entries.length} files)
            </p>
          )}

          {/* Summary */}
          <div className="mt-3 pt-3 border-t border-[#e9d5ff] grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <Summary label="Pairs" value={String(info.pairCount)} highlight={info.pairCount > 0} />
            <Summary
              label="Format"
              value={info.sampleExt ? info.sampleExt.toUpperCase().slice(1) : '—'}
            />
            <Summary
              label="Dimensions"
              value={info.sampleDims ? `${info.sampleDims.w} × ${info.sampleDims.h}` : '—'}
            />
            <Summary label="Total size" value={formatBytes(info.totalBytes)} />
          </div>

          {info.pairCount === 0 && (
            <p className="mt-3 text-xs text-[#D97706]">
              ⚠ No matched <code>src</code>↔<code>dst</code> pairs. Filenames must match (e.g. <code>frame_001.exr</code> in both folders).
            </p>
          )}
          {info.pairCount > 0 && info.unmatchedSrc.length > 0 && (
            <div className="mt-3 text-xs text-[#D97706]">
              <p>
                ⚠ {info.unmatchedSrc.length} src file{info.unmatchedSrc.length === 1 ? '' : 's'} have no matching dst
                {' '}(skipped during training):
              </p>
              <ul className="mt-1 max-h-32 overflow-auto list-disc pl-5 font-mono text-[11px] text-[#92400E]">
                {info.unmatchedSrc.slice(0, 100).map(n => <li key={n}>{n}</li>)}
              </ul>
              {info.unmatchedSrc.length > 100 && (
                <p className="mt-1">…and {info.unmatchedSrc.length - 100} more.</p>
              )}
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="text-xs text-[#EF4444]">{error}</p>
      )}
    </div>
  )
}

function Summary({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-[#9ca3af] font-semibold">{label}</p>
      <p className={`mt-0.5 ${highlight ? 'font-bold text-[#7E3AF2]' : 'text-[#374151]'}`}>{value}</p>
    </div>
  )
}

// ── helpers ─────────────────────────────────────────────────────────────────

function extractExt(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot < 0 ? '' : name.slice(dot).toLowerCase()
}

function stripExt(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot < 0 ? name : name.slice(0, dot)
}

function decodeDimensions(file: File): Promise<{ w: number; h: number } | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload  = () => { URL.revokeObjectURL(url); resolve({ w: img.naturalWidth, h: img.naturalHeight }) }
    img.onerror = () => { URL.revokeObjectURL(url); resolve(null) }
    img.src = url
  })
}

function formatBytes(b: number): string {
  if (b < 1024)             return `${b} B`
  if (b < 1024 * 1024)      return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
}
