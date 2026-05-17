import { redirect } from 'next/navigation'

export default async function JobDetailRedirect({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  redirect(`/demo/jobs/${id}`)
}
