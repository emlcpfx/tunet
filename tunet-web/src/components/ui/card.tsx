import { type HTMLAttributes } from 'react'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  hover?: boolean
}

export function Card({ hover, className = '', children, ...props }: CardProps) {
  return (
    <div
      className={`
        bg-white border border-[#e5e7eb] rounded-xl p-5 card-shadow
        ${hover ? 'transition-transform duration-200 hover:scale-[1.02] cursor-pointer' : ''}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className = '', children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`mb-4 ${className}`} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ className = '', children, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={`text-base font-semibold text-[#111827] ${className}`} {...props}>
      {children}
    </h3>
  )
}

interface MetricCardProps {
  label: string
  value: string
  sub?: string
  accent?: boolean
}

export function MetricCard({ label, value, sub, accent }: MetricCardProps) {
  return (
    <Card>
      <p className="text-xs text-[#6b7280] mb-1">{label}</p>
      <p className={`text-3xl font-bold ${accent ? 'text-[#ae69f4]' : 'text-[#111827]'}`}>{value}</p>
      {sub && <p className="text-xs text-[#9ca3af] mt-1">{sub}</p>}
    </Card>
  )
}
