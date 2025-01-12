<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <h2 class="text-lg font-semibold text-gray-900 mb-4">Index Connection</h2>
    
    <div class="flex gap-3">
      <input
        v-model="serverUrl"
        type="text"
        placeholder="http://localhost:8765 or /ipfs/Qm..."
        :disabled="isConnected || isLoading"
        class="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all disabled:bg-gray-100 disabled:text-gray-500"
        @keyup.enter="handleConnect"
      />
      <button
        v-if="!isConnected"
        @click="handleConnect"
        :disabled="isLoading || !serverUrl.trim()"
        class="px-6 py-2.5 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        <span v-if="isLoading" class="flex items-center gap-2">
          <svg class="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
          </svg>
          Connecting...
        </span>
        <span v-else>Connect</span>
      </button>
      <button
        v-else
        @click="handleDisconnect"
        class="px-6 py-2.5 bg-gray-600 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors"
      >
        Disconnect
      </button>
    </div>

    <div v-if="error" class="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
      {{ error }}
    </div>

    <div v-if="isConnected && indexInfo" class="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
      <div class="flex items-center gap-2 text-green-700 font-medium mb-2">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
        </svg>
        Connected
        <span v-if="connectionType" class="ml-2 px-2 py-0.5 text-xs rounded-full" 
              :class="connectionType === 'ipfs' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'">
          {{ connectionType.toUpperCase() }}
        </span>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <span class="text-gray-500">Documents:</span>
          <span class="ml-1 font-medium text-gray-900">{{ indexInfo.numDocs.toLocaleString() }}</span>
        </div>
        <div>
          <span class="text-gray-500">Segments:</span>
          <span class="ml-1 font-medium text-gray-900">{{ indexInfo.numSegments }}</span>
        </div>
        <div class="col-span-2">
          <span class="text-gray-500">Default fields:</span>
          <span class="ml-1 font-medium text-gray-900">{{ indexInfo.defaultFields.join(', ') || 'none' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  isLoading: Boolean,
  isConnected: Boolean,
  connectionType: String,
  error: String,
  indexInfo: Object
})

const emit = defineEmits(['connect', 'disconnect'])

const serverUrl = ref('http://localhost:8765')

const handleConnect = () => {
  if (serverUrl.value.trim()) {
    emit('connect', serverUrl.value.trim())
  }
}

const handleDisconnect = () => {
  emit('disconnect')
}
</script>
