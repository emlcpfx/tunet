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
}

export default nextConfig
