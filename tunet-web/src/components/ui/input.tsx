import { type InputHTMLAttributes, type TextareaHTMLAttributes, forwardRef } from 'react'

const inputClass = `
  w-full border border-[#e5e7eb] rounded-lg px-3.5 py-2.5 text-sm text-[#111827]
  bg-white placeholder-[#9ca3af] font-[family-name:var(--font-sans)]
  transition-colors duration-150
  focus:outline-none focus:border-[#ae69f4] focus:ring-3 focus:ring-[#ae69f4]/10
  disabled:bg-[#F9FAFB] disabled:text-[#9ca3af] disabled:cursor-not-allowed
`.trim()

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className = '', ...props }, ref) => (
    <input ref={ref} className={`${inputClass} ${className}`} {...props} />
  ),
)
Input.displayName = 'Input'

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className = '', ...props }, ref) => (
    <textarea ref={ref} className={`${inputClass} resize-none ${className}`} {...props} />
  ),
)
Textarea.displayName = 'Textarea'

interface SelectProps extends InputHTMLAttributes<HTMLSelectElement> {
  children: React.ReactNode
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className = '', children, ...props }, ref) => (
    <select ref={ref} className={`${inputClass} cursor-pointer ${className}`} {...props}>
      {children}
    </select>
  ),
)
Select.displayName = 'Select'

interface FormRowProps {
  label: string
  hint?: string
  /**
   * Multiline explainer shown on hover/focus of an info icon next to the
   * label. For artist-friendly "what does this actually do" text. Use \n
   * for line breaks — they render preserved.
   */
  tip?: string
  children: React.ReactNode
  required?: boolean
}

export function FormRow({ label, hint, tip, children, required }: FormRowProps) {
  return (
    <div className="space-y-1.5">
      <label className="flex items-center gap-1.5 text-sm font-medium text-[#374151]">
        <span>
          {label}
          {required && <span className="text-[#EF4444] ml-0.5">*</span>}
        </span>
        {tip && <InfoTip text={tip} />}
      </label>
      {children}
      {hint && <p className="text-xs text-[#9ca3af]">{hint}</p>}
    </div>
  )
}

/**
 * Tiny "(?)" info icon with a hover/focus popover. Multiline `text` is
 * preserved via whitespace-pre-wrap. Positioned absolutely so it doesn't
 * push other layout around.
 *
 * Why a custom thing instead of a real tooltip lib: zero deps, and the
 * desktop tunet.py tooltips are rich multiline blocks that don't fit a
 * generic "title=…" attribute.
 */
export function InfoTip({ text, align = 'left' }: { text: string; align?: 'left' | 'right' }) {
  return (
    <span className="relative inline-flex group" tabIndex={0}>
      <svg
        width="13" height="13" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2"
        className="text-[#9ca3af] hover:text-[#7E3AF2] cursor-help transition-colors"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
      <span
        role="tooltip"
        className={`
          pointer-events-none invisible opacity-0
          group-hover:visible group-hover:opacity-100
          group-focus-within:visible group-focus-within:opacity-100
          transition-opacity duration-100 delay-150
          absolute ${align === 'right' ? 'right-0' : 'left-0'} top-5 z-50
          bg-[#1F2937] text-[#F3F4F6] text-xs
          rounded-md shadow-lg px-3 py-2
          w-72 max-w-[18rem]
          whitespace-pre-wrap font-normal leading-relaxed
        `}
      >
        {text}
      </span>
    </span>
  )
}
