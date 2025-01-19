<template>
  <div class="bg-gray-900 rounded-xl p-6 text-white">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold flex items-center gap-2">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
        </svg>
        Network Stats
      </h2>
      <button
        @click="$emit('reset')"
        class="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
      >
        Reset
      </button>
    </div>

    <div v-if="!stats" class="text-gray-400 text-sm">
      No network activity yet
    </div>

    <template v-else>
      <div class="grid grid-cols-3 gap-6 mb-6">
        <div>
          <div class="text-3xl font-bold">{{ stats.total_requests }}</div>
          <div class="text-xs text-gray-400 uppercase tracking-wide">Requests</div>
        </div>
        <div>
          <div class="text-3xl font-bold">{{ formatBytes(stats.total_bytes) }}</div>
          <div class="text-xs text-gray-400 uppercase tracking-wide">Transferred</div>
        </div>
        <div>
          <div class="text-3xl font-bold">{{ totalTime }}ms</div>
          <div class="text-xs text-gray-400 uppercase tracking-wide">Total Time</div>
        </div>
      </div>

      <div v-if="cacheStats" class="mb-6 p-3 bg-gray-800 rounded-lg">
        <div class="text-xs text-gray-400 uppercase tracking-wide mb-1">Cache</div>
        <div class="text-sm">
          {{ formatBytes(cacheStats.total_bytes) }} / {{ formatBytes(cacheStats.max_bytes) }}
          <span class="text-gray-400">({{ cacheStats.total_slices }} slices, {{ cacheStats.files_cached }} files)</span>
        </div>
      </div>

      <div v-if="stats.requests?.length" ref="requestsContainer" class="space-y-2 max-h-48 overflow-y-auto">
        <div
          v-for="(req, idx) in stats.requests"
          :key="idx"
          class="flex items-center justify-between text-sm py-2 border-b border-gray-800 last:border-0"
        >
          <div class="flex items-center gap-2 min-w-0">
            <span class="text-green-400 font-mono">GET</span>
            <span v-if="req.range_start != null" class="text-orange-400 font-mono text-xs">
              [{{ req.range_start }}-{{ req.range_end }}]
            </span>
            <span v-else class="text-green-400 font-mono text-xs">[full]</span>
            <span class="text-gray-300 truncate">{{ getFileName(req.path) }}</span>
          </div>
          <div class="text-gray-500 text-xs shrink-0 ml-2">
            {{ formatBytes(req.bytes) }}
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, ref, watch, nextTick } from 'vue'

const props = defineProps({
  stats: Object,
  cacheStats: Object
})

defineEmits(['reset'])

const requestsContainer = ref(null)

// Auto-scroll to bottom when new requests appear
watch(
  () => props.stats?.requests?.length,
  () => {
    nextTick(() => {
      if (requestsContainer.value) {
        requestsContainer.value.scrollTop = requestsContainer.value.scrollHeight
      }
    })
  }
)

const totalTime = computed(() => {
  if (!props.stats?.operations) return 0
  return props.stats.operations.reduce((sum, op) => sum + op.duration_ms, 0)
})

const formatBytes = (bytes) => {
  if (!bytes) return '0 B'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

const getFileName = (path) => {
  if (!path) return ''
  // Extract filename from IPFS path like /ipfs/Qm.../filename.ext
  const parts = path.split('/')
  return parts[parts.length - 1] || path
}
</script>
