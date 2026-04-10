'use client'
import { type ButtonHTMLAttributes, forwardRef } from 'react'

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost'
type Size    = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  loading?: boolean
}

const base = 'inline-flex items-center justify-center gap-2 font-semibold rounded-lg transition-all duration-150 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2'

const variants: Record<Variant, string> = {
  primary:   'bg-[#ae69f4] text-white hover:bg-[#7E3AF2] active:bg-[#6C2BD9] focus-visible:ring-[#ae69f4]',
  secondary: 'bg-transparent text-[#ae69f4] border border-[#ae69f4] hover:bg-[#F7F4FC] focus-visible:ring-[#ae69f4]',
  danger:    'bg-[#EF4444] text-white hover:bg-[#DC2626] focus-visible:ring-[#EF4444]',
  ghost:     'bg-transparent text-[#6b7280] hover:bg-[#F9FAFB] hover:text-[#374151] focus-visible:ring-[#ae69f4]',
}

const sizes: Record<Size, string> = {
  sm:  'px-3 py-1.5 text-xs',
  md:  'px-4 py-2.5 text-sm',
  lg:  'px-5 py-3 text-sm',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'primary', size = 'md', loading, children, className = '', disabled, ...props }, ref) => (
    <button
      ref={ref}
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <svg className="animate-spin -ml-1 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {children}
    </button>
  ),
)
Button.displayName = 'Button'
