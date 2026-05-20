import type { DefaultSession } from 'next-auth'

declare module 'next-auth' {
  interface Session {
    user: DefaultSession['user'] & { id: string }
    // Set when a Keycloak refresh fails — the UI should bounce to /sign-in.
    error?: 'RefreshAccessTokenError'
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    // Keycloak tokens. The access token is Spark-capable (same realm) and is
    // forwarded to the Spark API server-side. It is intentionally NOT exposed
    // on the Session object, so it never reaches the browser.
    accessToken?: string
    refreshToken?: string
    accessTokenExpires?: number
    error?: 'RefreshAccessTokenError'
  }
}
