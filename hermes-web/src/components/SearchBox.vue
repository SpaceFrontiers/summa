<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <h2 class="text-lg font-semibold text-gray-900 mb-4">Search</h2>
    
    <div class="flex gap-3">
      <input
        v-model="query"
        type="text"
        placeholder="Enter search query..."
        :disabled="!isConnected || isSearching"
        class="flex-1 px-4 py-3 text-lg border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all disabled:bg-gray-100"
        @keyup.enter="handleSearch"
      />
      <input
        v-model.number="limit"
        type="number"
        min="1"
        max="100"
        :disabled="!isConnected || isSearching"
        class="w-20 px-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none disabled:bg-gray-100"
      />
      <button
        @click="handleSearch"
        :disabled="!isConnected || isSearching || !query.trim()"
        class="px-8 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
      >
        <span v-if="isSearching" class="flex items-center gap-2">
          <svg class="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
          </svg>
          Searching...
        </span>
        <span v-else>Search</span>
      </button>
    </div>

    <p v-if="!isConnected" class="mt-2 text-sm text-gray-500">
      Connect to a server first to enable search
    </p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  isConnected: Boolean,
  isSearching: Boolean
})

const emit = defineEmits(['search'])

const query = ref('')
const limit = ref(10)

const handleSearch = () => {
  if (query.value.trim()) {
    emit('search', { query: query.value.trim(), limit: limit.value })
  }
}
</script>
