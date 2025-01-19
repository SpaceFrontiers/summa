import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useAppConfig } from '../composables/useAppConfig'

const STORAGE_KEY = 'hermes-connections'
const MAX_RECENT = 10

// Well-known index location (relative to current origin)
const WELL_KNOWN_INDEX = '/index'

/**
 * Store for managing connection history and recent URLs
 *
 * Each connection can have:
 * - url: The connection URL
 * - label: Display label (from IPNS name or ux.dsl short_name)
 */
export const useConnectionsStore = defineStore('connections', () => {
  // Recent connections (most recent first)
  // Each item: { url: string, label?: string }
  const recentConnections = ref([])

  // Current connection URL
  const currentUrl = ref(null)

  // Current connection label
  const currentLabel = ref(null)

  // Computed: recent URLs for backward compatibility
  const recentUrls = computed(() => recentConnections.value.map(c => c.url))

  // Check if well-known index exists
  const hasWellKnownIndex = ref(false)

  // Load from localStorage on init
  const loadFromStorage = () => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const data = JSON.parse(stored)
        // Support both old format (recentUrls array of strings) and new format
        if (data.recentConnections) {
          recentConnections.value = data.recentConnections
        } else if (data.recentUrls) {
          // Migrate from old format
          recentConnections.value = data.recentUrls.map(url => ({ url }))
        }
        currentUrl.value = data.currentUrl || null
        currentLabel.value = data.currentLabel || null
      }
    } catch (e) {
      console.warn('Failed to load connections from localStorage:', e)
    }
  }

  // Save to localStorage
  const saveToStorage = () => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        recentConnections: recentConnections.value,
        currentUrl: currentUrl.value,
        currentLabel: currentLabel.value
      }))
    } catch (e) {
      console.warn('Failed to save connections to localStorage:', e)
    }
  }

  // Add a connection to recent list
  const addRecent = (url, label = null) => {
    if (!url) return

    // Remove if already exists (to move to top)
    const index = recentConnections.value.findIndex(c => c.url === url)
    if (index !== -1) {
      // Update label if provided
      if (label) {
        recentConnections.value[index].label = label
      }
      recentConnections.value.splice(index, 1)
    }

    // Add to beginning
    recentConnections.value.unshift({ url, label })

    // Limit size
    if (recentConnections.value.length > MAX_RECENT) {
      recentConnections.value = recentConnections.value.slice(0, MAX_RECENT)
    }

    saveToStorage()
  }

  // Update label for a connection
  const updateLabel = (url, label) => {
    const conn = recentConnections.value.find(c => c.url === url)
    if (conn) {
      conn.label = label
      saveToStorage()
    }
    if (currentUrl.value === url) {
      currentLabel.value = label
      saveToStorage()
    }
  }

  // Set current connection
  const setCurrentUrl = (url, label = null) => {
    currentUrl.value = url
    currentLabel.value = label
    if (url) {
      addRecent(url, label)
    }
    saveToStorage()
  }

  // Remove a URL from recent list
  const removeRecent = (url) => {
    const index = recentConnections.value.findIndex(c => c.url === url)
    if (index !== -1) {
      recentConnections.value.splice(index, 1)
      saveToStorage()
    }
  }

  // Clear all recent connections
  const clearRecent = () => {
    recentConnections.value = []
    saveToStorage()
  }

  // Disconnect (clear current but keep history)
  const disconnect = () => {
    currentUrl.value = null
    currentLabel.value = null
    saveToStorage()
  }

  // Well-known index label (from ux.dsl short_name or title)
  const wellKnownIndexLabel = ref(null)

  // Check for well-known index at /index and fetch its label
  const checkWellKnownIndex = async () => {
    try {
      // Get base URL for IPFS gateways
      const appConfig = useAppConfig()
      if (!appConfig.isDetected.value) {
        appConfig.detect()
      }
      const baseUrl = appConfig.getBaseUrl() // Returns '' for non-IPFS, or '/ipfs/CID' prefix for IPFS gateways

      const response = await fetch(`${baseUrl}${WELL_KNOWN_INDEX}/schema.json`, {
        method: 'HEAD',
        signal: AbortSignal.timeout(5000)
      })
      hasWellKnownIndex.value = response.ok

      if (response.ok) {
        // Try to fetch ux.dsl to get the label
        try {
          const dslResponse = await fetch(`${baseUrl}${WELL_KNOWN_INDEX}/ux.dsl`, {
            signal: AbortSignal.timeout(3000)
          })
          if (dslResponse.ok) {
            const text = await dslResponse.text()
            // Simple extraction of short_name or title
            const shortNameMatch = text.match(/short_name\s+"([^"]+)"/)
            const titleMatch = text.match(/title\s+"([^"]+)"/)
            wellKnownIndexLabel.value = shortNameMatch?.[1] || titleMatch?.[1] || null
          }
        } catch {
          // ux.dsl is optional
        }
      }

      return response.ok
    } catch {
      hasWellKnownIndex.value = false
      return false
    }
  }

  // Get display label for a URL
  const getDisplayLabel = (url) => {
    // Check if we have a stored label
    const conn = recentConnections.value.find(c => c.url === url)
    if (conn?.label) return conn.label

    // For IPNS, extract the name
    if (url.startsWith('/ipns/') || url.startsWith('ipns://')) {
      const name = url.replace(/^(\/ipns\/|ipns:\/\/)/, '').split('/')[0]
      return name
    }

    // For well-known index
    if (url === WELL_KNOWN_INDEX) {
      return wellKnownIndexLabel.value || 'Local Index'
    }

    return null
  }

  // Initialize
  loadFromStorage()

  return {
    recentConnections,
    recentUrls,
    currentUrl,
    currentLabel,
    hasWellKnownIndex,
    wellKnownIndex: WELL_KNOWN_INDEX,
    wellKnownIndexLabel,
    addRecent,
    updateLabel,
    setCurrentUrl,
    removeRecent,
    clearRecent,
    disconnect,
    checkWellKnownIndex,
    getDisplayLabel
  }
})
