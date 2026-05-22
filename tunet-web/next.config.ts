import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  output: 'standalone',
  // Production deploys (deploy/deploy.ps1) build into a separate distDir so
  // `next build` never clobbers a running `next dev` server's .next/ — the two
  // sharing one directory corrupts dev's manifests (ENOENT _buildManifest.tmp).
  // Dev keeps the default .next; deploy sets TUNET_DIST_DIR=.next-prod.
  distDir: process.env.TUNET_DIST_DIR || '.next',
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '*.proxy.runpod.net',
      },
      {
        protocol: 'https',
        hostname: '*.supabase.co',
      },
    ],
  },
  // Allow large form-data uploads to /api/spark/upload-stage — multi-hundred-MB
  // batches of training frames.
  experimental: {
    serverActions: {
      bodySizeLimit: '500mb',
    },
  },
  // The app moved off the /demo/* prefix to clean root paths (dashboard at /).
  // Keep old bookmarks/links working. Evaluated in order — the dashboard and the
  // bare /demo land on / before the catch-all maps the rest 1:1.
  async redirects() {
    return [
      { source: '/demo', destination: '/', permanent: false },
      { source: '/demo/dashboard', destination: '/', permanent: false },
      { source: '/demo/:path*', destination: '/:path*', permanent: false },
    ]
  },
}

export default nextConfig
