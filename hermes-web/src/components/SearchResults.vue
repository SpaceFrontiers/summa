<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-gray-900">Results</h2>
      <div v-if="results" class="text-sm text-gray-500">
        {{ currentOffset + 1 }}-{{ currentOffset + results.hits.length }}
      </div>
    </div>

    <div v-if="!results" class="text-center py-12 text-gray-500">
      Enter a search query to see results
    </div>

    <div v-else-if="results.hits.length === 0" class="text-center py-12 text-gray-500">
      No results found
    </div>

    <div v-else class="space-y-4">
      <div
        v-for="(hit, idx) in results.hits"
        :key="`${hit.address.segment_id}-${hit.address.doc_id}`"
        class="border border-gray-200 rounded-lg overflow-hidden p-4"
        :class="[uxConfig?.hasRowClick?.value ? 'cursor-pointer hover:border-blue-300' : '', uxConfig?.config?.value?.styles?.result_card || '']"
        @click="handleRowClick(idx)"
      >
        <!-- Document content -->
        <div v-if="hit.document" class="space-y-3">
            <!-- Use UX config columns if available -->
            <template v-if="displayColumns.length > 0">
              <template
                v-for="fieldName in displayColumns"
                :key="fieldName"
              >
              <div v-if="!isFieldEmpty(hit.document[fieldName])"
              >
                <!-- Label (skip for ipfs_files which handles its own label) -->
                <div v-if="getFieldLabel(fieldName) && getRenderType(fieldName) !== 'ipfs_files'" class="text-xs font-medium text-gray-400 uppercase tracking-wide mb-0.5">
                  {{ getFieldLabel(fieldName) }}
                </div>
                <!-- Special render types -->
                <template v-if="getRenderType(fieldName) === 'uri_links'">
                  <div class="flex flex-wrap gap-2">
                    <a
                      v-for="(link, linkIdx) in formatField(fieldName, hit.document[fieldName]).links"
                      :key="linkIdx"
                      :href="link.url"
                      target="_blank"
                      class="text-sm text-blue-600 hover:underline"
                    >
                      {{ link.label }}
                    </a>
                  </div>
                </template>
                <template v-else-if="getRenderType(fieldName) === 'split_newline'">
                  <div class="text-sm text-gray-900">
                    <div v-for="(line, lineIdx) in formatField(fieldName, hit.document[fieldName]).lines" :key="lineIdx" :class="lineIdx === 0 ? 'font-semibold' : 'text-gray-600'">
                      {{ line }}
                    </div>
                  </div>
                </template>
                <template v-else-if="getRenderType(fieldName) === 'ipfs_files'">
                  <IpfsFileLinks
                    :id="String(hit.document[fieldName])"
                    :formats="getFieldConfig(fieldName).ipfs_formats || ['pdf', 'epub', 'djvu']"
                    :base="getFieldConfig(fieldName).ipfs_base || ''"
                    :label="getFieldLabel(fieldName)"
                  />
                </template>
                <!-- Default text render -->
                <div
                  v-else
                  class="text-sm text-gray-900"
                  :class="[getFieldStyles(fieldName), isFieldClickable(fieldName) ? 'cursor-pointer hover:underline' : '']"
                  @click="handleFieldClick(fieldName, hit.document)"
                >
                  {{ formatField(fieldName, hit.document[fieldName]) }}
                </div>
              </div>
              </template>
            </template>
            <!-- Fallback: show all fields -->
            <template v-else>
              <div
                v-for="(value, key) in hit.document"
                :key="key"
              >
                <div class="text-xs font-medium text-gray-400 uppercase tracking-wide mb-0.5">{{ key }}</div>
                <div class="text-sm text-gray-900">
                  {{ typeof value === 'object' ? JSON.stringify(value) : value }}
                </div>
              </div>
            </template>
        </div>

        <!-- No stored fields -->
        <div v-else class="text-sm text-gray-500">
          No stored fields
        </div>
      </div>
    </div>

    <!-- Pagination controls -->
    <div v-if="results && results.hits.length > 0" class="mt-4 pt-4 border-t border-gray-200 flex items-center justify-between">
      <button
        @click="$emit('prev-page')"
        :disabled="currentOffset === 0"
        class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        ← Previous
      </button>
      <span class="text-sm text-gray-500">
        Page {{ Math.floor(currentOffset / pageSize) + 1 }}
      </span>
      <button
        @click="$emit('next-page')"
        :disabled="results.hits.length < pageSize"
        class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Next →
      </button>
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
import { ref, computed } from 'vue'
import IpfsFileLinks from './IpfsFileLinks.vue'

const props = defineProps({
  results: Object,
  uxConfig: Object,
  fieldNames: {
    type: Array,
    default: () => []
  },
  currentOffset: {
    type: Number,
    default: 0
  },
  pageSize: {
    type: Number,
    default: 5
  }
})

const emit = defineEmits(['next-page', 'prev-page'])

const showRaw = ref(false)

// Check if a field value is empty (null, undefined, empty string, empty array, 0)
const isFieldEmpty = (value) => {
  if (value == null) return true
  if (value === '') return true
  if (value === 0) return true
  if (Array.isArray(value) && value.length === 0) return true
  return false
}

// Computed display columns from UX config or all fields
const displayColumns = computed(() => {
  if (props.uxConfig?.getDisplayColumns) {
    return props.uxConfig.getDisplayColumns(props.fieldNames)
  }
  return props.fieldNames
})

// UX config helper methods with fallbacks
const getFieldLabel = (fieldName) => {
  if (props.uxConfig?.getFieldLabel) {
    return props.uxConfig.getFieldLabel(fieldName)
  }
  return fieldName.charAt(0).toUpperCase() + fieldName.slice(1).replace(/_/g, ' ')
}

const formatField = (fieldName, value) => {
  if (props.uxConfig?.formatField) {
    return props.uxConfig.formatField(fieldName, value)
  }
  if (value === undefined || value === null) return ''
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

const getFieldStyles = (fieldName) => {
  if (props.uxConfig?.getFieldStyles) {
    return props.uxConfig.getFieldStyles(fieldName)
  }
  return ''
}

const getRenderType = (fieldName) => {
  if (props.uxConfig?.getFieldRenderType) {
    return props.uxConfig.getFieldRenderType(fieldName)
  }
  return 'text'
}

const getFieldConfig = (fieldName) => {
  if (props.uxConfig?.getFieldConfig) {
    return props.uxConfig.getFieldConfig(fieldName)
  }
  return {}
}

const isFieldClickable = (fieldName) => {
  if (props.uxConfig?.isFieldClickable) {
    return props.uxConfig.isFieldClickable(fieldName)
  }
  return false
}

const handleFieldClick = (fieldName, doc) => {
  if (props.uxConfig?.handleFieldClick) {
    props.uxConfig.handleFieldClick(fieldName, doc)
  }
}

const handleRowClick = (idx) => {
  const hit = props.results?.hits?.[idx]
  if (props.uxConfig?.hasRowClick?.value && hit?.document) {
    props.uxConfig.handleRowClick(hit.document)
  }
}

</script>
