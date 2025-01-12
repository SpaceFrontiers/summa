<template>
  <div class="min-h-screen bg-gray-50">
    <div class="max-w-4xl mx-auto px-4 py-8">
      <header class="mb-8">
        <h1 class="text-3xl font-bold text-gray-900 flex items-center gap-3">
          <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.3-4.3"/>
          </svg>
          Hermes Search
        </h1>
        <p class="text-gray-500 mt-1">WASM-powered search engine</p>
      </header>

      <div class="space-y-6">
        <ServerConnect
          :is-loading="hermes.isLoading.value"
          :is-connected="hermes.isConnected.value"
          :connection-type="hermes.connectionType.value"
          :error="hermes.error.value"
          :index-info="hermes.indexInfo.value"
          @connect="handleConnect"
          @disconnect="handleDisconnect"
        />

        <SearchBox
          :is-connected="hermes.isConnected.value"
          :is-searching="isSearching"
          @search="handleSearch"
        />

        <SearchResults
          :results="searchResults"
          @load-document="handleLoadDocument"
        />

        <NetworkStats
          :stats="hermes.networkStats.value"
          :cache-stats="hermes.cacheStats.value"
          @reset="hermes.resetNetworkStats"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useHermes } from './composables/useHermes'
import ServerConnect from './components/ServerConnect.vue'
import SearchBox from './components/SearchBox.vue'
import SearchResults from './components/SearchResults.vue'
import NetworkStats from './components/NetworkStats.vue'

const hermes = useHermes()
const isSearching = ref(false)
const searchResults = ref(null)

const handleConnect = async (serverUrl) => {
  await hermes.connect(serverUrl)
}

const handleDisconnect = () => {
  hermes.disconnect()
  searchResults.value = null
}

const handleSearch = async ({ query, limit }) => {
  isSearching.value = true
  try {
    searchResults.value = await hermes.search(query, limit)
  } catch (e) {
    console.error('Search error:', e)
  } finally {
    isSearching.value = false
  }
}

const handleLoadDocument = async ({ segmentId, docId, callback }) => {
  try {
    const doc = await hermes.getDocument(segmentId, docId)
    callback(doc)
  } catch (e) {
    console.error('Load document error:', e)
    callback(null)
  }
}
</script>
