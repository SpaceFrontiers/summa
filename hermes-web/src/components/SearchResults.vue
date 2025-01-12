<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-gray-900">Results</h2>
      <div v-if="results" class="text-sm text-gray-500">
        {{ results.total_hits }} hits
      </div>
    </div>

    <div v-if="!results" class="text-center py-12 text-gray-500">
      Enter a search query to see results
    </div>

    <div v-else-if="results.hits.length === 0" class="text-center py-12 text-gray-500">
      No results found
    </div>

    <div v-else class="space-y-3">
      <div
        v-for="(hit, idx) in results.hits"
        :key="`${hit.address.segment_id}-${hit.address.doc_id}`"
        class="border border-gray-200 rounded-lg overflow-hidden"
      >
        <div
          class="flex items-center justify-between px-4 py-3 bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
          @click="toggleExpand(idx)"
        >
          <div class="flex items-center gap-3">
            <span class="text-sm font-mono text-gray-600">
              #{{ idx + 1 }}
            </span>
            <span class="text-sm text-gray-900">
              {{ hit.address.segment_id.slice(0, 8) }}...{{ hit.address.doc_id }}
            </span>
          </div>
          <div class="flex items-center gap-3">
            <span class="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded">
              Score: {{ hit.score.toFixed(4) }}
            </span>
            <svg
              class="w-5 h-5 text-gray-400 transition-transform"
              :class="{ 'rotate-180': expandedItems.has(idx) }"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
            </svg>
          </div>
        </div>

        <div v-if="expandedItems.has(idx)" class="p-4 border-t border-gray-200">
          <div v-if="loadingDocs.has(idx)" class="flex items-center justify-center py-8">
            <svg class="animate-spin h-6 w-6 text-blue-600" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            <span class="ml-2 text-gray-500">Loading document...</span>
          </div>

          <div v-else-if="loadedDocs[idx]" class="space-y-2">
            <div
              v-for="(value, key) in loadedDocs[idx]"
              :key="key"
              class="flex gap-2"
            >
              <span class="text-sm font-medium text-gray-500 shrink-0">{{ key }}:</span>
              <span class="text-sm text-gray-900 break-all">
                {{ typeof value === 'object' ? JSON.stringify(value) : value }}
              </span>
            </div>
          </div>

          <div v-else class="text-sm text-gray-500">
            No stored fields
          </div>
        </div>
      </div>
    </div>

    <div v-if="results" class="mt-4 pt-4 border-t border-gray-200">
      <button
        @click="showRaw = !showRaw"
        class="text-sm text-gray-500 hover:text-gray-700"
      >
        {{ showRaw ? 'Hide' : 'Show' }} raw response
      </button>
      <pre
        v-if="showRaw"
        class="mt-2 p-4 bg-gray-900 text-green-400 rounded-lg text-xs overflow-auto max-h-64"
      >{{ JSON.stringify(results, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch } from 'vue'

const props = defineProps({
  results: Object
})

const emit = defineEmits(['loadDocument'])

const expandedItems = ref(new Set())
const loadingDocs = ref(new Set())
const loadedDocs = reactive({})
const showRaw = ref(false)

watch(() => props.results, () => {
  expandedItems.value = new Set()
  loadingDocs.value = new Set()
  Object.keys(loadedDocs).forEach(key => delete loadedDocs[key])
})

const toggleExpand = async (idx) => {
  if (expandedItems.value.has(idx)) {
    expandedItems.value.delete(idx)
    expandedItems.value = new Set(expandedItems.value)
  } else {
    expandedItems.value.add(idx)
    expandedItems.value = new Set(expandedItems.value)
    
    if (!loadedDocs[idx] && !loadingDocs.value.has(idx)) {
      loadingDocs.value.add(idx)
      loadingDocs.value = new Set(loadingDocs.value)
      
      const hit = props.results.hits[idx]
      emit('loadDocument', {
        idx,
        segmentId: hit.address.segment_id,
        docId: hit.address.doc_id,
        callback: (doc) => {
          loadedDocs[idx] = doc
          loadingDocs.value.delete(idx)
          loadingDocs.value = new Set(loadingDocs.value)
        }
      })
    }
  }
}
</script>
