'use client'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input, FormRow, Select } from '@/components/ui/input'
import { TierBadge } from '@/components/ui/badge'
import { FileUpload } from '@/components/dashboard/file-upload'
import { CREDIT_PACKS, formatCredits } from '@/types'
import type { DbGpuPricing } from '@/types'

async function uploadFile(file: File, role: string, jobId: string): Promise<string | null> {
  // 1. Get presigned upload URL
  const res = await fetch('/api/upload/presign', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileName: file.name, fileType: file.type, jobId, role }),
  })
  if (!res.ok) return null
  const { signedUrl, path } = await res.json()

  // 2. Upload directly to Supabase Storage
  const uploadRes = await fetch(signedUrl, {
    method: 'PUT',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  })
  if (!uploadRes.ok) return null
  return path as string
}

export default function NewJobPage() {
  const router = useRouter()
  const [gpuList, setGpuList] = useState<DbGpuPricing[]>([])
  const [selectedGpu, setSelectedGpu] = useState<DbGpuPricing | null>(null)
  const [name, setName] = useState('')
  const [containerDisk, setContainerDisk] = useState(50)
  const [volumeGb, setVolumeGb] = useState(100)
  const [configFile, setConfigFile] = useState<File | null>(null)
  const [srcFile, setSrcFile]         = useState<File | null>(null)
  const [dstFile, setDstFile]         = useState<File | null>(null)
  const [checkpointFile, setCheckpointFile] = useState<File | null>(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [balance, setBalance]         = useState(0)

  useEffect(() => {
    fetch('/api/gpu-pricing').then(r => r.json()).then((data: DbGpuPricing[]) => {
      setGpuList(data)
      const rec = data.find(g => g.tier === 'recommended') ?? data[0]
      if (rec) setSelectedGpu(rec)
    })
    fetch('/api/billing/balance').then(r => r.json()).then(d => setBalance(d.balance_cents ?? 0))
  }, [])

  const canAfford = selectedGpu
    ? balance >= Math.ceil(selectedGpu.platform_cost_per_hr * 100)
    : false

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!selectedGpu || !configFile) return
    setLoading(true)
    setError(null)

    try {
      // Create a provisional job ID for scoped storage paths
      const jobId = crypto.randomUUID()

      // Upload files in parallel
      const [configPath, srcPath, dstPath, checkpointPath] = await Promise.all([
        uploadFile(configFile, 'config', jobId),
        srcFile ? uploadFile(srcFile, 'src', jobId) : Promise.resolve(null),
        dstFile ? uploadFile(dstFile, 'dst', jobId) : Promise.resolve(null),
        checkpointFile ? uploadFile(checkpointFile, 'checkpoint', jobId) : Promise.resolve(null),
      ])

      if (!configPath) throw new Error('Config upload failed')

      // Create job + launch pod
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: jobId,
          name: name || `${selectedGpu.short_key}-${new Date().toISOString().slice(0, 10)}`,
          gpu_type_id: selectedGpu.gpu_type_id,
          config_path: configPath,
          src_zip_path: srcPath,
          dst_zip_path: dstPath,
          checkpoint_path: checkpointPath,
          container_disk_gb: containerDisk,
          volume_gb: volumeGb,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.error ?? 'Failed to launch job')
      }

      const { id } = await res.json()
      router.push(`/jobs/${id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Launch failed')
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 animate-slide-in max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-[#111827]">New Training Job</h1>
        <p className="text-sm text-[#6b7280] mt-1">Configure and launch a GPU training run</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* GPU Selection */}
        <Card>
          <CardTitle className="mb-4">GPU</CardTitle>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {gpuList.map(gpu => (
              <button
                key={gpu.gpu_type_id}
                type="button"
                onClick={() => setSelectedGpu(gpu)}
                className={`text-left p-4 rounded-xl border-2 transition-all duration-150 ${
                  selectedGpu?.gpu_type_id === gpu.gpu_type_id
                    ? 'border-[#ae69f4] bg-[#F7F4FC]'
                    : 'border-[#e5e7eb] hover:border-[#D1D5DB]'
                }`}
              >
                <div className="flex items-start justify-between mb-1">
                  <span className="font-semibold text-sm text-[#111827]">{gpu.display_name}</span>
                  <TierBadge tier={gpu.tier} />
                </div>
                <p className="text-xs text-[#6b7280]">{gpu.vram_gb} GB VRAM</p>
                <p className="text-sm font-semibold text-[#ae69f4] mt-2">
                  ${gpu.platform_cost_per_hr.toFixed(2)}<span className="text-xs font-normal text-[#9ca3af]">/hr</span>
                </p>
                {selectedGpu?.gpu_type_id === gpu.gpu_type_id && (
                  <div className="flex gap-3 mt-2 text-xs text-[#6b7280]">
                    <span>1h: ${gpu.platform_cost_per_hr.toFixed(2)}</span>
                    <span>4h: ${(gpu.platform_cost_per_hr * 4).toFixed(2)}</span>
                    <span>8h: ${(gpu.platform_cost_per_hr * 8).toFixed(2)}</span>
                  </div>
                )}
              </button>
            ))}
          </div>

          {selectedGpu && !canAfford && (
            <div className="mt-3 p-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-xs text-[#EF4444]">
              Insufficient credits. You need at least {formatCredits(Math.ceil(selectedGpu.platform_cost_per_hr * 100))} to launch.{' '}
              <a href="/billing" className="underline font-medium">Top up</a>
            </div>
          )}
        </Card>

        {/* Job Config */}
        <Card>
          <CardTitle className="mb-4">Job Configuration</CardTitle>
          <div className="space-y-4">
            <FormRow label="Job Name" hint="Leave blank to auto-generate">
              <Input
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder={selectedGpu ? `${selectedGpu.short_key}-job` : 'my-training-job'}
              />
            </FormRow>

            <FormRow label="Config YAML" required hint="Your tunet config.yaml">
              <FileUpload
                accept=".yaml,.yml"
                label="YAML config file"
                maxMb={1}
                onFile={setConfigFile}
                currentName={configFile?.name}
              />
            </FormRow>

            <div className="grid grid-cols-2 gap-4">
              <FormRow label="Src Images" hint="ZIP of source images (optional if path is in config)">
                <FileUpload
                  accept=".zip"
                  label="src.zip"
                  maxMb={10000}
                  onFile={setSrcFile}
                  currentName={srcFile?.name}
                />
              </FormRow>
              <FormRow label="Dst Images" hint="ZIP of destination images (optional)">
                <FileUpload
                  accept=".zip"
                  label="dst.zip"
                  maxMb={10000}
                  onFile={setDstFile}
                  currentName={dstFile?.name}
                />
              </FormRow>
            </div>

            <FormRow label="Resume Checkpoint" hint=".pth file to resume from (optional)">
              <FileUpload
                accept=".pth"
                label=".pth checkpoint file"
                maxMb={10000}
                onFile={setCheckpointFile}
                currentName={checkpointFile?.name}
              />
            </FormRow>
          </div>
        </Card>

        {/* Storage */}
        <Card>
          <CardTitle className="mb-4">Storage</CardTitle>
          <div className="grid grid-cols-2 gap-4">
            <FormRow label="Container Disk" hint="OS + dependencies">
              <Select value={containerDisk} onChange={e => setContainerDisk(Number(e.target.value))}>
                {[20, 50, 100, 200].map(v => (
                  <option key={v} value={v}>{v} GB</option>
                ))}
              </Select>
            </FormRow>
            <FormRow label="Persistent Volume" hint="For checkpoints and outputs">
              <Select value={volumeGb} onChange={e => setVolumeGb(Number(e.target.value))}>
                {[50, 100, 200, 500].map(v => (
                  <option key={v} value={v}>{v} GB</option>
                ))}
              </Select>
            </FormRow>
          </div>
        </Card>

        {error && (
          <div className="p-3 bg-[#FEF2F2] border border-[#fecaca] rounded-lg text-sm text-[#EF4444]">
            {error}
          </div>
        )}

        {/* Submit */}
        <div className="flex items-center justify-between pt-2">
          <div className="text-sm text-[#6b7280]">
            Balance: <span className="font-semibold text-[#111827]">{formatCredits(balance)}</span>
          </div>
          <div className="flex gap-3">
            <Button type="button" variant="ghost" onClick={() => router.back()}>Cancel</Button>
            <Button
              type="submit"
              loading={loading}
              disabled={!selectedGpu || !configFile || !canAfford}
            >
              Launch Job
            </Button>
          </div>
        </div>
      </form>
    </div>
  )
}
