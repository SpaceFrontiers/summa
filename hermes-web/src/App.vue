<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Top Bar (always visible) -->
    <TopBar
      :title="hermes.uxConfig?.title?.value || 'Hermes Search'"
      :short-title="hermes.uxConfig?.short_name?.value || 'Hermes'"
      :logo="hermes.uxConfig?.logo?.value"
      :is-connected="hermes.isConnected.value"
      :connection-type="hermes.connectionType.value"
      :current-url="hermes.currentUrl.value"
      :index-info="hermes.indexInfo.value"
      :network-stats="hermes.networkStats.value"
      :cache-stats="hermes.cacheStats.value"
      @disconnect="handleDisconnect"
      @connect="handleConnect"
      @show-connect="showConnectModal = true"
    />

    <!-- Connect Page (when not connected) -->
    <ConnectPage
      v-if="!hermes.isConnected.value"
      :is-loading="hermes.isLoading.value"
      :error="hermes.error.value"
      :progress="hermes.connectionProgress.value"
      :download-stats="hermes.downloadStats"
      @connect="handleConnect"
      @abort="hermes.abortConnect"
    />

    <!-- Main Search UI (when connected) -->
    <div v-else class="max-w-4xl mx-auto px-4 py-6">
      <div class="space-y-6">
        <SearchBox
          :is-connected="hermes.isConnected.value"
          :is-searching="isSearching"
          :placeholder="hermes.uxConfig?.placeholder?.value"
          @search="handleSearch"
        />

        <!-- Searching spinner -->
        <div v-if="isSearching" class="flex items-center justify-center py-12">
          <svg class="animate-spin h-8 w-8 text-blue-600" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
          </svg>
          <span class="ml-3 text-gray-500">Searching...</span>
        </div>

        <!-- Results (only show after search) -->
        <SearchResults
          v-else-if="searchResults"
          :results="searchResults"
          :ux-config="hermes.uxConfig"
          :field-names="hermes.indexInfo.value?.fieldNames || []"
          :current-offset="currentOffset"
          :page-size="pageSize"
          @load-document="handleLoadDocument"
          @next-page="handleNextPage"
          @prev-page="handlePrevPage"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useHermes } from './composables/useHermes'
import { useConnectionsStore } from './stores/connections'
import TopBar from './components/TopBar.vue'
import ConnectPage from './components/ConnectPage.vue'
import SearchBox from './components/SearchBox.vue'
import SearchResults from './components/SearchResults.vue'

const hermes = useHermes()
const connectionsStore = useConnectionsStore()
const isSearching = ref(false)
const searchResults = ref(null)
const showConnectModal = ref(false)

// Auto-connect to last used engine on startup
onMounted(() => {
  const lastUrl = connectionsStore.currentUrl
  if (lastUrl && !hermes.isConnected.value) {
    handleConnect(lastUrl)
  }
})

const handleConnect = async (serverUrl) => {
  const success = await hermes.connect(serverUrl)
  if (success) {
    connectionsStore.setCurrentUrl(serverUrl)
  }
}

const handleDisconnect = () => {
  hermes.disconnect()
  connectionsStore.disconnect()
  searchResults.value = null
}

const currentQuery = ref('')
const currentOffset = ref(0)
const pageSize = 5

const handleSearch = async ({ query, limit, offset = 0 }) => {
  isSearching.value = true
  currentQuery.value = query
  currentOffset.value = offset
  try {
    searchResults.value = await hermes.search(query, limit || pageSize, offset)
  } catch (e) {
    console.error('Search error:', e)
  } finally {
    isSearching.value = false
  }
}

const handleNextPage = () => {
  if (searchResults.value && searchResults.value.hits.length === pageSize) {
    handleSearch({ query: currentQuery.value, limit: pageSize, offset: currentOffset.value + pageSize })
  }
}

const handlePrevPage = () => {
  if (currentOffset.value > 0) {
    handleSearch({ query: currentQuery.value, limit: pageSize, offset: Math.max(0, currentOffset.value - pageSize) })
  }
}

const handleLoadDocument = async ({ segmentId, docId, callback }) => {
  try {
    const doc = await hermes.getDocument(segmentId, docId)
    console.log('Loaded document:', segmentId, docId, doc)
    callback(doc)
  } catch (e) {
    console.error('Load document error:', e)
    callback(null)
  }
}
</script>
