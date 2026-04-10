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
  children: React.ReactNode
  required?: boolean
}

export function FormRow({ label, hint, children, required }: FormRowProps) {
  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-[#374151]">
        {label}
        {required && <span className="text-[#EF4444] ml-0.5">*</span>}
      </label>
      {children}
      {hint && <p className="text-xs text-[#9ca3af]">{hint}</p>}
    </div>
  )
}
