'use client'
import { useEffect, useRef } from 'react'

interface LogViewerProps {
  lines: string[]
  autoScroll?: boolean
  height?: string
}

export function LogViewer({ lines, autoScroll = true, height = '240px' }: LogViewerProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight
    }
  }, [lines, autoScroll])

  return (
    <div
      ref={ref}
      className="log-console overflow-y-auto"
      style={{ height }}
    >
      {lines.length === 0 ? (
        <span className="text-[#6b7280]">Waiting for logs...</span>
      ) : (
        lines.map((line, i) => (
          <div key={i} className="whitespace-pre-wrap break-all leading-5">
            {line}
          </div>
        ))
      )}
    </div>
  )
}
