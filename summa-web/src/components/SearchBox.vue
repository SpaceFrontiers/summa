<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-3 sm:p-6">
    <h2 class="text-lg font-semibold text-gray-900 mb-4">Search</h2>

    <div class="flex gap-2 sm:gap-3">
      <input
        v-model="query"
        type="text"
        :placeholder="placeholder || 'Enter search query...'"
        :disabled="!isConnected || isSearching"
        class="flex-1 min-w-0 px-3 sm:px-4 py-2.5 sm:py-3 text-base sm:text-lg border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all disabled:bg-gray-100"
        @keyup.enter="handleSearch"
      />
      <button
        @click="handleSearch"
        :disabled="!isConnected || isSearching || !query.trim()"
        class="px-4 sm:px-8 py-2.5 sm:py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shrink-0"
      >
        Search
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
  isSearching: Boolean,
  placeholder: String
})

const emit = defineEmits(['search'])

const query = ref('')

const handleSearch = () => {
  if (query.value.trim()) {
    emit('search', { query: query.value.trim(), limit: 5 })
  }
}
</script>
