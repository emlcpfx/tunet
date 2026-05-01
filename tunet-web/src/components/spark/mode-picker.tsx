'use client'

/**
 * Mode picker — top-level choice between New / Resume / Fine-tune.
 *
 * Drives the rest of the new-job form:
 *   New      — train from scratch, free choice of preset + arch.
 *   Resume   — continue a prior job. Output dir is forced to the source's,
 *              preset/arch are locked (train.py rejects mismatches).
 *   Fine-tune — start from a prior job's weights but with a fresh optimizer.
 *               Permissive on architecture; new output dir.
 */

import type { TrainingMode } from '@/lib/spark-form-state'

interface ModePickerProps {
  value:    TrainingMode
  onChange: (m: TrainingMode) => void
}

const OPTIONS: {
  key:        TrainingMode
  label:      string
  blurb:      string
  icon:       React.ReactNode
}[] = [
  {
    key:   'new',
    label: 'New',
    blurb: 'Train from scratch with full control over preset and model.',
    icon:  (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
    ),
  },
  {
    key:   'resume',
    label: 'Resume',
    blurb: 'Continue a prior job from its latest checkpoint. Same preset, same data, more steps.',
    icon:  (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 4 23 10 17 10" />
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
      </svg>
    ),
  },
  {
    key:   'finetune',
    label: 'Fine-tune',
    blurb: 'Start from a prior job’s weights with new data. Optimizer resets to step 0.',
    icon:  (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  },
]

export function ModePicker({ value, onChange }: ModePickerProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3" role="radiogroup" aria-label="Training mode">
      {OPTIONS.map(o => {
        const selected = o.key === value
        return (
          <button
            key={o.key}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(o.key)}
            className={`text-left p-4 rounded-lg border-2 transition-all ${
              selected
                ? 'border-[#ae69f4] bg-[#F7F4FC]'
                : 'border-[#e5e7eb] bg-white hover:border-[#D1D5DB]'
            }`}
          >
            <div className={`flex items-center gap-2 mb-1.5 ${selected ? 'text-[#7E3AF2]' : 'text-[#374151]'}`}>
              {o.icon}
              <span className="font-semibold text-sm">{o.label}</span>
              {selected && (
                <svg className="ml-auto" width="16" height="16" viewBox="0 0 24 24" fill="#ae69f4">
                  <path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                </svg>
              )}
            </div>
            <p className="text-xs text-[#6b7280] leading-relaxed">{o.blurb}</p>
          </button>
        )
      })}
    </div>
  )
}
