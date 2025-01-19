<template>
  <header class="bg-white border-b border-gray-200 sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-4">
      <div class="flex items-center justify-between h-14">
        <!-- Logo and Title -->
        <div class="flex items-center gap-3">
          <img v-if="logo" :src="logo" class="w-7 h-7" alt="Logo" />
          <svg v-else class="w-7 h-7 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <span class="text-lg font-semibold text-gray-900">
            <span class="hidden sm:inline">{{ title }}</span>
            <span class="sm:hidden">{{ shortTitle }}</span>
          </span>
        </div>

        <!-- Right side: Stats + Connection -->
        <div class="flex items-center gap-3">
          <!-- Network Stats (compact) -->
          <div v-if="isConnected && networkStats" class="relative">
            <button
              @click="showStats = !showStats"
              class="flex items-center gap-1.5 px-2 py-1 text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-50 rounded transition-colors"
              :title="`${networkStats.total_requests} requests, ${formatBytes(networkStats.total_bytes)}`"
            >
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
              <span class="font-mono">{{ formatBytes(networkStats.total_bytes) }}</span>
            </button>

            <!-- Stats dropdown -->
            <div
              v-if="showStats"
              class="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-2 text-xs"
            >
              <div class="px-3 py-1 text-gray-500 font-medium">Network Stats</div>
              <div class="px-3 py-1 flex justify-between">
                <span class="text-gray-600">Requests:</span>
                <span class="font-mono text-gray-900">{{ networkStats.total_requests }}</span>
              </div>
              <div class="px-3 py-1 flex justify-between">
                <span class="text-gray-600">Downloaded:</span>
                <span class="font-mono text-gray-900">{{ formatBytes(networkStats.total_bytes) }}</span>
              </div>
              <div v-if="cacheStats" class="border-t border-gray-100 mt-1 pt-1">
                <div class="px-3 py-1 text-gray-500 font-medium">Cache</div>
                <div class="px-3 py-1 flex justify-between">
                  <span class="text-gray-600">Size:</span>
                  <span class="font-mono text-gray-900">{{ formatBytes(cacheStats.total_bytes) }}</span>
                </div>
                <div class="px-3 py-1 flex justify-between">
                  <span class="text-gray-600">Slices:</span>
                  <span class="font-mono text-gray-900">{{ cacheStats.total_slices }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Connection Status / Dropdown -->
        <div class="relative">
          <button
            v-if="isConnected"
            @click="showDropdown = !showDropdown"
            class="flex items-center gap-2 px-3 py-1.5 text-sm bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors"
          >
            <span class="w-2 h-2 bg-green-500 rounded-full"></span>
            <span class="hidden sm:inline truncate max-w-32">{{ displayUrl }}</span>
            <span class="sm:hidden">Connected</span>
            <svg class="w-4 h-4" :class="{ 'rotate-180': showDropdown }" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <button
            v-else
            @click="$emit('showConnect')"
            class="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <span class="w-2 h-2 bg-gray-400 rounded-full"></span>
            Not connected
          </button>

          <!-- Dropdown Menu -->
          <div
            v-if="showDropdown && isConnected"
            class="absolute right-0 mt-2 w-72 bg-white rounded-lg shadow-lg border border-gray-200 py-2"
          >
            <div class="px-4 py-2 border-b border-gray-100">
              <div class="text-xs text-gray-500 mb-1">Connected to</div>
              <div class="text-sm font-medium text-gray-900 truncate">{{ currentUrl }}</div>
              <div v-if="connectionType" class="mt-1">
                <span class="px-2 py-0.5 text-xs rounded-full"
                      :class="connectionType === 'ipfs' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'">
                  {{ connectionType.toUpperCase() }}
                </span>
              </div>
            </div>

            <div v-if="indexInfo" class="px-4 py-2 border-b border-gray-100 text-xs text-gray-600">
              <div class="flex justify-between">
                <span>Documents:</span>
                <span class="font-medium">{{ indexInfo.numDocs?.toLocaleString() }}</span>
              </div>
              <div class="flex justify-between mt-1">
                <span>Segments:</span>
                <span class="font-medium">{{ indexInfo.numSegments }}</span>
              </div>
            </div>

            <!-- Recent connections -->
            <div v-if="recentConnections.length > 1" class="px-4 py-2 border-b border-gray-100">
              <div class="text-xs text-gray-500 mb-2">Switch to</div>
              <div class="space-y-1 max-h-32 overflow-y-auto">
                <button
                  v-for="conn in recentConnections.filter(c => c.url !== currentUrl).slice(0, 3)"
                  :key="conn.url"
                  @click="switchConnection(conn.url)"
                  class="w-full text-left text-sm text-gray-700 hover:bg-gray-50 px-2 py-1 rounded"
                >
                  <span v-if="conn.label || getDisplayLabel(conn.url)" class="font-medium block truncate">
                    {{ conn.label || getDisplayLabel(conn.url) }}
                  </span>
                  <span :class="conn.label || getDisplayLabel(conn.url) ? 'text-xs text-gray-400' : ''" class="truncate block">
                    {{ conn.url }}
                  </span>
                </button>
              </div>
            </div>

            <button
              @click="handleDisconnect"
              class="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50"
            >
              Disconnect
            </button>
          </div>
        </div>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useConnectionsStore } from '../stores/connections'

const connectionsStore = useConnectionsStore()
const recentConnections = connectionsStore.recentConnections
const getDisplayLabel = connectionsStore.getDisplayLabel

const props = defineProps({
  title: {
    type: String,
    default: 'Hermes Search'
  },
  shortTitle: {
    type: String,
    default: 'Hermes'
  },
  logo: String,
  isConnected: Boolean,
  connectionType: String,
  currentUrl: String,
  indexInfo: Object,
  networkStats: Object,
  cacheStats: Object
})

const emit = defineEmits(['disconnect', 'connect', 'showConnect'])

const showDropdown = ref(false)
const showStats = ref(false)

const formatBytes = (bytes) => {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const displayUrl = computed(() => {
  if (!props.currentUrl) return ''

  // Use stored label if available
  const label = getDisplayLabel(props.currentUrl)
  if (label) return label

  const url = props.currentUrl
  if (url.startsWith('/ipfs/') || url.startsWith('/ipns/')) {
    const parts = url.split('/')
    if (parts[2]) {
      const name = parts[2]
      return name.length > 20 ? name.slice(0, 10) + '...' + name.slice(-8) : name
    }
  }
  return url.replace(/^https?:\/\//, '').slice(0, 20)
})

const switchConnection = (url) => {
  showDropdown.value = false
  emit('connect', url)
}

const handleDisconnect = () => {
  showDropdown.value = false
  emit('disconnect')
}

const handleClickOutside = (e) => {
  if (!e.target.closest('.relative')) {
    showDropdown.value = false
    showStats.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>
