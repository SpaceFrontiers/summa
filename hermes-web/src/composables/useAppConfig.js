import { ref, readonly } from 'vue'

// Singleton state
const isIpfsGateway = ref(false)
const isLocalDaemon = ref(false)
const gatewayUrl = ref(null)
const localDaemonUrl = ref(null) // Base URL for local daemon (e.g., http://localhost:8080)
const ipfsCid = ref(null)
const isDetected = ref(false)

/**
 * App configuration composable
 *
 * Detects if the app is served from an IPFS gateway or local daemon
 * and provides configuration that can be reused across the application.
 */
export function useAppConfig() {

  /**
   * Detect if app is served from IPFS gateway
   *
   * IPFS gateways serve content at URLs like:
   * - https://<cid>.ipfs.dweb.link/
   * - https://ipfs.io/ipfs/<cid>/
   * - http://<cid>.ipfs.localhost:8080/
   */
  const detect = () => {
    if (isDetected.value) return

    const hostname = window.location.hostname
    const pathname = window.location.pathname
    const origin = window.location.origin

    // Check subdomain gateway format: <cid>.ipfs.<host>
    // Use window.location.host (includes port) not hostname (excludes port)
    const host = window.location.host // e.g., "bafyb4i....ipfs.localhost:8080"
    const subdomainMatch = host.match(/^([a-z0-9]+)\.ipfs\.(.+)$/i)
    if (subdomainMatch) {
      ipfsCid.value = subdomainMatch[1]
      gatewayUrl.value = origin
      isIpfsGateway.value = true
      const gatewayHost = subdomainMatch[2] // e.g., "localhost:8080" or "dweb.link"
      isLocalDaemon.value = gatewayHost.includes('localhost') || gatewayHost.includes('127.0.0.1')
      if (isLocalDaemon.value) {
        // Extract base daemon URL (e.g., http://localhost:8080)
        localDaemonUrl.value = `${window.location.protocol}//${gatewayHost}`
      }
      isDetected.value = true
      console.log(`App served from IPFS ${isLocalDaemon.value ? 'local daemon' : 'gateway'}: ${origin} (CID: ${ipfsCid.value})`)
      if (isLocalDaemon.value) {
        console.log(`Local daemon URL: ${localDaemonUrl.value}`)
      }
      return
    }

    // Check path gateway format: /ipfs/<cid>/
    const pathMatch = pathname.match(/^\/ipfs\/([a-z0-9]+)/i)
    if (pathMatch) {
      ipfsCid.value = pathMatch[1]
      gatewayUrl.value = origin
      isIpfsGateway.value = true
      isLocalDaemon.value = hostname === 'localhost' || hostname === '127.0.0.1'
      if (isLocalDaemon.value) {
        localDaemonUrl.value = origin
      }
      isDetected.value = true
      console.log(`App served from IPFS ${isLocalDaemon.value ? 'local daemon' : 'gateway'}: ${origin} (CID: ${ipfsCid.value})`)
      return
    }

    // Not served from IPFS
    isIpfsGateway.value = false
    isLocalDaemon.value = false
    gatewayUrl.value = null
    ipfsCid.value = null
    isDetected.value = true
    console.log('App served from regular HTTP server')
  }

  /**
   * Get the base URL for fetching resources
   * If served from IPFS, returns the gateway URL with CID
   */
  const getBaseUrl = () => {
    if (!isIpfsGateway.value) return ''

    const hostname = window.location.hostname

    // Subdomain format
    if (hostname.match(/^[a-z0-9]+\.ipfs\./i)) {
      return window.location.origin
    }

    // Path format
    return `${window.location.origin}/ipfs/${ipfsCid.value}`
  }

  /**
   * Convert a relative path to an IPFS path if served from IPFS
   */
  const toIpfsPath = (relativePath) => {
    if (!isIpfsGateway.value || !ipfsCid.value) return relativePath

    const cleanPath = relativePath.startsWith('/') ? relativePath.slice(1) : relativePath
    return `/ipfs/${ipfsCid.value}/${cleanPath}`
  }

  return {
    // State (readonly)
    isIpfsGateway: readonly(isIpfsGateway),
    isLocalDaemon: readonly(isLocalDaemon),
    gatewayUrl: readonly(gatewayUrl),
    localDaemonUrl: readonly(localDaemonUrl),
    ipfsCid: readonly(ipfsCid),
    isDetected: readonly(isDetected),

    // Methods
    detect,
    getBaseUrl,
    toIpfsPath
  }
}
