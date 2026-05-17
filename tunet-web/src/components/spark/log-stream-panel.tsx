'use client'
import { useEffect, useRef, useState } from 'react'

interface LogStreamPanelProps {
  jobId: string
  initiallyLive: boolean
  /** Called once when the first log line arrives (parent can collapse provisioning UI) */
  onFirstLine?: () => void
}

/**
 * Streams SSE from /api/spark/jobs/[id]/logs and renders into a scroll-locked
 * dark panel. The server proxy (route handler) replays the entire log from
 * the start of the job, then live-tails — so this component just reads.
 *
 * Spark's actual SSE shape (verified live):
 *
 *   event: log
 *   id: <iso-ts>-<stream>
 *   data: {"ts":"2026-05-01T16:43:25.375Z",
 *          "stream":"stdout"|"stderr",
 *          "phase":"agent"|"container",
 *          "line":"the actual log content"}
 *
 *   event: status     (may also be sent with status updates)
 *   data: {...}
 *
 * We listen on the named 'log' event (NOT default 'message'), parse the JSON,
 * extract `.line`, and tag stderr lines with a red gutter.
 */

interface LogLine {
  ts:     string
  stream: 'stdout' | 'stderr'
  phase:  'agent' | 'container'
  line:   string
}

export function LogStreamPanel({ jobId, initiallyLive, onFirstLine }: LogStreamPanelProps) {
  const [lines, setLines] = useState<LogLine[]>([])
  const [state, setState] = useState<'connecting' | 'streaming' | 'closed' | 'error'>('connecting')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [showAgent, setShowAgent] = useState(true)   // toggle agent vs container output
  const [copyLabel, setCopyLabel] = useState<'Copy' | 'Copied!'>('Copy')
  const containerRef = useRef<HTMLDivElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const firstLineFiredRef = useRef(false)

  useEffect(() => {
    const url = `/api/spark/jobs/${jobId}/logs`
    const es  = new EventSource(url)

    es.onopen = () => setState('streaming')

    const handleLogEvent = (ev: MessageEvent) => {
      let parsed: LogLine
      try {
        parsed = JSON.parse(ev.data) as LogLine
      } catch {
        // Fallback: treat as plain text
        parsed = { ts: '', stream: 'stdout', phase: 'container', line: ev.data }
      }

      setLines(prev => prev.length > 5000
        ? [...prev.slice(prev.length - 4000), parsed]
        : [...prev, parsed]
      )
      if (!firstLineFiredRef.current) {
        firstLineFiredRef.current = true
        onFirstLine?.()
      }
    }

    // Spark uses named 'log' events. Some implementations also fire the default
    // 'message' event — listen on both to be safe.
    es.addEventListener('log', handleLogEvent as EventListener)
    es.addEventListener('message', handleLogEvent as EventListener)

    es.addEventListener('error', (ev) => {
      // SSE errors are opaque on the client. Differentiate "closed naturally"
      // from "actually broken" by readyState.
      if (es.readyState === EventSource.CLOSED) {
        setState('closed')
      } else {
        setState('error')
        const data = (ev as MessageEvent).data
        if (typeof data === 'string') setErrorMsg(data)
      }
    })

    return () => es.close()
  }, [jobId])

  const visibleLines = showAgent ? lines : lines.filter(l => l.phase === 'container')

  async function copyVisible() {
    if (visibleLines.length === 0) return
    const text = visibleLines.map(l => l.line).join('\n')
    try {
      await navigator.clipboard.writeText(text)
      setCopyLabel('Copied!')
      setTimeout(() => setCopyLabel('Copy'), 1500)
    } catch {
      // Fallback for non-secure contexts: a hidden textarea + execCommand
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      try { document.execCommand('copy'); setCopyLabel('Copied!'); setTimeout(() => setCopyLabel('Copy'), 1500) }
      catch { /* truly no clipboard — give up silently */ }
      document.body.removeChild(ta)
    }
  }

  // Auto-scroll to bottom when new lines arrive (unless user scrolled up)
  useEffect(() => {
    if (!autoScroll || !scrollRef.current) return
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [lines, autoScroll])

  // Detect manual scroll-up to disable auto-scroll
  function onScroll() {
    if (!scrollRef.current) return
    const el = scrollRef.current
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    setAutoScroll(atBottom)
  }

  const stateLabel = (
    state === 'connecting' ? 'Connecting…' :
    state === 'streaming'  ? (initiallyLive ? 'Streaming' : 'Replay') :
    state === 'closed'     ? 'Stream closed' :
    'Stream error'
  )

  const stateColor = (
    state === 'streaming' && initiallyLive ? 'text-[#16A34A]' :
    state === 'error'                      ? 'text-[#EF4444]' :
    state === 'closed'                     ? 'text-[#9ca3af]' :
                                             'text-[#1c64f2]'
  )

  return (
    <div className="bg-[#0f172a] border border-[#1e293b] rounded-lg overflow-hidden flex flex-col" ref={containerRef}>
      {/* toolbar */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#0b1220] border-b border-[#1e293b] text-xs">
        <span className={`font-mono ${stateColor}`}>● {stateLabel}</span>
        <div className="flex items-center gap-3">
          <span className="text-[#475569]">
            {visibleLines.length}{showAgent ? '' : ` / ${lines.length}`} lines
          </span>
          <label className="flex items-center gap-1.5 text-[#64748b] cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showAgent}
              onChange={(e) => setShowAgent(e.target.checked)}
              className="accent-[#ae69f4]"
            />
            Show agent
          </label>
          <button
            onClick={copyVisible}
            disabled={visibleLines.length === 0}
            title={`Copy ${visibleLines.length} line${visibleLines.length === 1 ? '' : 's'} to clipboard`}
            className="flex items-center gap-1 text-[#64748b] hover:text-[#cbd5e1] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
            </svg>
            {copyLabel}
          </button>
          <button
            onClick={() => setLines([])}
            className="text-[#64748b] hover:text-[#cbd5e1] transition-colors"
          >
            Clear
          </button>
          <label className="flex items-center gap-1.5 text-[#64748b] cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="accent-[#ae69f4]"
            />
            Auto-scroll
          </label>
        </div>
      </div>

      {/* scrollable log body */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="overflow-auto px-3 py-2 font-mono text-[12px] leading-relaxed text-[#cdd6f4] whitespace-pre-wrap break-words"
        style={{ height: '480px' }}
      >
        {visibleLines.length === 0 && state === 'connecting' && (
          <p className="text-[#475569] italic">Connecting to log stream…</p>
        )}
        {visibleLines.length === 0 && state === 'streaming' && (
          <p className="text-[#475569] italic">
            {lines.length > 0 ? 'No container output yet (only agent log so far) — toggle "Show agent" to see what\'s happening.' : 'Waiting for first log line…'}
          </p>
        )}
        {visibleLines.map((l, i) => (
          <LogLineRow key={i} line={l} />
        ))}
        {state === 'error' && errorMsg && (
          <p className="text-[#EF4444] italic mt-2">[stream error] {errorMsg}</p>
        )}
        {state === 'closed' && (
          <p className="text-[#475569] italic mt-2">[stream closed]</p>
        )}
      </div>
    </div>
  )
}

function LogLineRow({ line }: { line: LogLine }) {
  // Color rules:
  //   stderr → red gutter
  //   agent  → dim purple prefix to distinguish from container output
  //   stdout container → default
  const isStderr   = line.stream === 'stderr'
  const isAgent    = line.phase === 'agent'

  return (
    <div className="flex gap-2 hover:bg-white/5">
      {isAgent && (
        <span className="text-[#7c3aed] flex-shrink-0 select-none">spark│</span>
      )}
      <span className={`flex-1 ${isStderr ? 'text-[#fca5a5]' : ''}`}>
        {line.line || ' '}
      </span>
    </div>
  )
}
