/**
 * Client-side registry of "hidden" job IDs.
 *
 * Spark Compute v1 doesn't expose a delete/archive endpoint — jobs stay in
 * the account's history forever. To keep the UI tidy we hide them locally
 * via localStorage. Hiding is reversible: a "Show hidden (N)" affordance on
 * the list/dashboard reveals them and a Restore button removes from the set.
 *
 * Shape: a flat string array (no schema versioning — this is throwaway client
 * state that's safe to drop on browser data clear).
 */

const STORAGE_KEY = 'tunet:hiddenJobs:v1'

export function readHidden(): Set<string> {
  if (typeof window === 'undefined') return new Set()
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return new Set()
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return new Set()
    return new Set(arr.filter((x): x is string => typeof x === 'string'))
  } catch {
    return new Set()
  }
}

function write(set: Set<string>) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(set)))
    // Notify other tabs/components on the same page
    window.dispatchEvent(new CustomEvent('tunet:hidden-jobs-changed'))
  } catch {
    // localStorage full / blocked — silently no-op
  }
}

export function hideJobs(ids: string[]) {
  const set = readHidden()
  for (const id of ids) set.add(id)
  write(set)
}

export function restoreJobs(ids: string[]) {
  const set = readHidden()
  for (const id of ids) set.delete(id)
  write(set)
}

export function clearAllHidden() {
  write(new Set())
}

/**
 * React hook that returns the current hidden set and re-renders when it
 * changes (across tabs or other components on the same page).
 */
import { useEffect, useState } from 'react'

export function useHiddenJobs(): Set<string> {
  const [hidden, setHidden] = useState<Set<string>>(() => readHidden())

  useEffect(() => {
    const refresh = () => setHidden(readHidden())
    window.addEventListener('storage',                    refresh)
    window.addEventListener('tunet:hidden-jobs-changed',  refresh)
    return () => {
      window.removeEventListener('storage',                    refresh)
      window.removeEventListener('tunet:hidden-jobs-changed',  refresh)
    }
  }, [])

  return hidden
}
