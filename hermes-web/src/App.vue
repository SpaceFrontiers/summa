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

        <!-- Search error -->
        <div
          v-else-if="searchError"
          class="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm"
        >
          <div class="flex items-start">
            <svg class="h-5 w-5 text-red-400 mr-2 mt-0.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clip-rule="evenodd" />
            </svg>
            <div>
              <div class="font-medium">Search failed</div>
              <div class="mt-1">{{ searchError }}</div>
            </div>
          </div>
        </div>

        <!-- Results (only show after search) -->
        <SearchResults
          v-else-if="searchResults"
          :results="searchResults"
          :ux-config="hermes.uxConfig"
          :field-names="hermes.indexInfo.value?.fieldNames || []"
          :current-offset="currentOffset"
          :page-size="pageSize"
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
const searchError = ref(null)
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
  searchError.value = null
}

const currentQuery = ref('')
const currentOffset = ref(0)
const pageSize = 5

const handleSearch = async ({ query, limit, offset = 0 }) => {
  isSearching.value = true
  searchError.value = null
  currentQuery.value = query
  currentOffset.value = offset
  try {
    searchResults.value = await hermes.search(query, limit || pageSize, offset)
  } catch (e) {
    console.error('Search error:', e)
    searchError.value = e.message || String(e)
    searchResults.value = null
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

</script>
