<template>
  <div class="min-h-[80vh] flex items-center justify-center">
    <div class="w-full max-w-md">
      <div class="text-center mb-8">
        <div class="flex justify-center mb-4">
          <svg class="w-16 h-16 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
        </div>
        <h1 class="text-3xl font-bold text-gray-900">Hermes Search</h1>
        <p class="text-gray-500 mt-2">Connect to a search index to get started</p>
      </div>

      <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Index URL</label>
            <input
              v-model="serverUrl"
              type="text"
              placeholder="http://localhost:8765 or /ipns/..."
              :disabled="isLoading"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all disabled:bg-gray-100"
              @keyup.enter="handleConnect"
              @focus="showRecent = true"
            />
          </div>

          <!-- Well-known index -->
          <div v-if="hasWellKnownIndex" class="space-y-2">
            <div class="text-xs text-gray-500">Available index</div>
            <button
              :disabled="isLoading"
              @click="selectUrl(wellKnownIndex)"
              class="w-full text-left px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
              <span class="font-medium text-blue-700">{{ wellKnownIndexLabel || 'Local Index' }}</span>
              <span class="text-blue-500 text-xs ml-auto">{{ wellKnownIndex }}</span>
            </button>
          </div>

          <!-- Recent connections -->
          <div v-if="recentConnections.length > 0" class="space-y-2">
            <div class="text-xs text-gray-500">Recent connections</div>
            <div class="space-y-1">
              <button
                v-for="conn in recentConnections.slice(0, 5)"
                :key="conn.url"
                :disabled="isLoading"
                @click="selectUrl(conn.url)"
                class="w-full text-left px-3 py-2 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span v-if="conn.label || getDisplayLabel(conn.url)" class="font-medium block">
                  {{ conn.label || getDisplayLabel(conn.url) }}
                </span>
                <span :class="conn.label || getDisplayLabel(conn.url) ? 'text-xs text-gray-400' : ''" class="truncate block">
                  {{ conn.url }}
                </span>
              </button>
            </div>
          </div>

          <div class="flex gap-2">
            <button
              @click="handleConnect"
              :disabled="isLoading || !serverUrl.trim()"
              class="flex-1 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <span v-if="isLoading" class="flex items-center justify-center gap-2">
                <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Connecting...
              </span>
              <span v-else>Connect</span>
            </button>
            <button
              v-if="isLoading"
              @click="emit('abort')"
              class="px-4 py-3 bg-gray-200 text-gray-700 font-medium rounded-lg hover:bg-gray-300 transition-colors"
              title="Cancel connection"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <!-- Progress status (unified: status message or download progress) -->
          <div v-if="isLoading" class="text-center text-xs text-gray-500 font-mono tabular-nums">
              <!-- Show download progress if available -->
              <template v-if="downloadStats?.value && (downloadStats.value.bytesDownloaded > 0 || downloadStats.value.currentDownload)">
                <span v-if="downloadStats.value.currentDownload" class="block text-gray-600">
                  {{ downloadStats.value.currentDownload.file }}
                  <span v-if="downloadStats.value.currentDownload.totalBytes">
                    · <span class="inline-block min-w-16 text-right">{{ formatBytes(downloadStats.value.currentDownload.bytesDownloaded) }}</span> / {{ formatBytes(downloadStats.value.currentDownload.totalBytes) }}
                    (<span class="inline-block w-8 text-right">{{ Math.round(downloadStats.value.currentDownload.bytesDownloaded / downloadStats.value.currentDownload.totalBytes * 100) }}%</span>)
                  </span>
                  <span v-else>
                    · {{ formatBytes(downloadStats.value.currentDownload.bytesDownloaded) }}
                  </span>
                </span>
                <span v-if="downloadStats.value.requestCount > 0" class="block text-gray-400">
                  {{ downloadStats.value.requestCount }} requests · {{ formatBytes(downloadStats.value.bytesDownloaded) }} total
                </span>
              </template>
              <!-- Otherwise show status message -->
              <span v-else-if="progress?.message" class="block text-gray-600">
                {{ progress.message }}
              </span>
          </div>

          <div v-if="error" class="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {{ error }}
          </div>
        </div>
      </div>

      <div class="text-center mt-6 text-sm text-gray-500">
        <p>Supports HTTP servers and IPFS/IPNS paths</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useConnectionsStore } from '../stores/connections'

defineProps({
  isLoading: Boolean,
  error: String,
  progress: Object,
  downloadStats: Object
})

const emit = defineEmits(['connect', 'abort'])

const connectionsStore = useConnectionsStore()
const serverUrl = ref('')
const showRecent = ref(false)

const recentConnections = connectionsStore.recentConnections
const hasWellKnownIndex = connectionsStore.hasWellKnownIndex
const wellKnownIndex = connectionsStore.wellKnownIndex
const wellKnownIndexLabel = connectionsStore.wellKnownIndexLabel
const getDisplayLabel = connectionsStore.getDisplayLabel

onMounted(async () => {
  // Check for well-known index first
  const indexExists = await connectionsStore.checkWellKnownIndex()

  if (connectionsStore.currentUrl) {
    serverUrl.value = connectionsStore.currentUrl
  } else if (indexExists) {
    // Pre-fill with /index if it exists
    serverUrl.value = wellKnownIndex
  } else if (recentConnections.value.length > 0) {
    serverUrl.value = recentConnections.value[0].url
  }
})

const handleConnect = () => {
  if (serverUrl.value.trim()) {
    connectionsStore.setCurrentUrl(serverUrl.value.trim())
    emit('connect', serverUrl.value.trim())
  }
}

const formatBytes = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const selectUrl = (url) => {
  serverUrl.value = url
  handleConnect()
}
</script>
