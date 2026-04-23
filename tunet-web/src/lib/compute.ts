/**
 * Provider abstraction — reads COMPUTE_PROVIDER env var and re-exports
 * the matching adapter. Job routes import from here, not from runpod.ts / spark.ts directly.
 *
 * COMPUTE_PROVIDER=runpod   (default)
 * COMPUTE_PROVIDER=spark
 */

const provider = process.env.COMPUTE_PROVIDER ?? 'runpod'

if (provider !== 'runpod' && provider !== 'spark') {
  throw new Error(`Unknown COMPUTE_PROVIDER: "${provider}". Must be "runpod" or "spark".`)
}

export type ComputeProvider = 'runpod' | 'spark'
export const ACTIVE_PROVIDER: ComputeProvider = provider as ComputeProvider

// These re-exports resolve at module load time. Next.js tree-shakes the unused branch.
export {
  createPod,
  getPod,
  stopPod,
  terminatePod,
  monitorBaseUrl,
  proxyMonitor,
  buildBootstrapScript,
} from './runpod'

// To switch to Spark, replace the line above with:
// } from './spark'
//
// Or make it dynamic once both providers are fully wired:
//   const mod = provider === 'spark' ? await import('./spark') : await import('./runpod')
