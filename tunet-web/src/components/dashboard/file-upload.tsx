'use client'
import { useRef, useState, type DragEvent } from 'react'

interface FileUploadProps {
  accept?: string
  label: string
  hint?: string
  maxMb?: number
  onFile: (file: File) => void
  currentName?: string
}

export function FileUpload({ accept, label, hint, maxMb, onFile, currentName }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleFile(file: File | null) {
    if (!file) return
    setError(null)
    if (maxMb && file.size > maxMb * 1024 * 1024) {
      setError(`File exceeds ${maxMb} MB limit`)
      return
    }
    onFile(file)
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0] ?? null)
  }

  return (
    <div className="space-y-1.5">
      <div
        className={`
          border-2 border-dashed rounded-xl p-6 text-center cursor-pointer
          transition-all duration-200
          ${dragging
            ? 'border-[#ae69f4] bg-[rgba(174,105,244,0.04)] scale-[1.01]'
            : 'border-[#D1D5DB] bg-[#F9FAFB] hover:border-[#ae69f4] hover:bg-[rgba(174,105,244,0.02)]'
          }
        `}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        <svg className="mx-auto mb-2 text-[#9ca3af]" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" x2="12" y1="3" y2="15"/>
        </svg>
        {currentName ? (
          <p className="text-sm font-medium text-[#ae69f4]">{currentName}</p>
        ) : (
          <p className="text-sm text-[#6b7280]">
            <span className="font-medium text-[#ae69f4]">Click to upload</span> or drag & drop
          </p>
        )}
        <p className="text-xs text-[#9ca3af] mt-1">{label}</p>
        {hint && <p className="text-xs text-[#9ca3af]">{hint}</p>}
      </div>
      {error && <p className="text-xs text-[#EF4444]">{error}</p>}
    </div>
  )
}
