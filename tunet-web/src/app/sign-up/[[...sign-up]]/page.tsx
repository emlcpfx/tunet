import { redirect } from 'next/navigation'

/** Registration is handled in Keycloak; reuse sign-in which starts the OIDC flow. */
export default function SignUpPage() {
  redirect('/sign-in')
}
