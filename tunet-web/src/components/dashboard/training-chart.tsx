'use client'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import type { MonitorMetrics } from '@/types'

interface TrainingChartProps {
  metrics: MonitorMetrics | null
  height?: number
}

export function TrainingChart({ metrics, height = 220 }: TrainingChartProps) {
  if (!metrics || metrics.steps.length === 0) {
    return (
      <div className="flex items-center justify-center h-[220px] text-sm text-[#9ca3af]">
        Waiting for training data...
      </div>
    )
  }

  // Build chart data — sample up to 500 points for performance
  const total = metrics.steps.length
  const step  = Math.max(1, Math.floor(total / 500))
  const data = metrics.steps
    .filter((_, i) => i % step === 0)
    .map((x, i) => {
      const idx = i * step
      const entry: Record<string, number | null> = {
        step: parseFloat(x.toFixed(2)),
        train: parseFloat(metrics.l1[idx]?.toFixed(4) ?? '0'),
      }
      if (metrics.lpips) {
        entry.lpips = parseFloat(metrics.lpips[idx]?.toFixed(4) ?? '0')
      }
      if (metrics.val_steps && metrics.val_l1) {
        // Find closest val entry
        const valIdx = metrics.val_steps.findIndex(vs => vs >= x)
        if (valIdx !== -1) {
          entry.val = parseFloat(metrics.val_l1[valIdx]?.toFixed(4) ?? '0')
        }
      }
      return entry
    })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
        <XAxis
          dataKey="step"
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#9ca3af' }}
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          tickLine={false}
          axisLine={false}
          width={48}
        />
        <Tooltip
          contentStyle={{
            background: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: 8,
            fontSize: 12,
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
          }}
          labelFormatter={(v) => `Epoch ${v}`}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          iconType="circle"
          iconSize={8}
        />
        <Line
          type="monotone"
          dataKey="train"
          name={`Train ${metrics.loss_label}`}
          stroke="#ae69f4"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 4 }}
        />
        {metrics.lpips && (
          <Line
            type="monotone"
            dataKey="lpips"
            name="LPIPS"
            stroke="#c084fc"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
          />
        )}
        {metrics.val_l1 && (
          <Line
            type="monotone"
            dataKey="val"
            name={`Val ${metrics.loss_label}`}
            stroke="#1c64f2"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
