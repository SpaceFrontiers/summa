<template>
  <!-- Only render if checking or has available formats -->
  <div v-if="isChecking || hasAvailableFormats">
    <div v-if="label" class="text-xs font-medium text-gray-400 uppercase tracking-wide mb-0.5">
      {{ label }}
    </div>
    <div class="flex flex-wrap gap-2">
    <template v-for="format in formats" :key="format">
      <a
        v-if="availableFormats[format]"
        :href="buildUrl(format)"
        target="_blank"
        class="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-md transition-colors"
        :class="formatClasses[format]"
      >
        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        {{ format.toUpperCase() }}
      </a>
      <span
        v-else-if="checking[format]"
        class="inline-flex items-center gap-1 px-2 py-1 text-xs text-gray-400 bg-gray-100 rounded-md"
      >
        <svg class="animate-spin w-3 h-3" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        {{ format.toUpperCase() }}
      </span>
    </template>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'

const props = defineProps({
  id: {
    type: String,
    required: true
  },
  formats: {
    type: Array,
    default: () => ['pdf', 'epub', 'djvu']
  },
  base: {
    type: String,
    required: true
  },
  label: {
    type: String,
    default: ''
  }
})

const availableFormats = reactive({})
const checking = reactive({})

const isChecking = computed(() => Object.values(checking).some(v => v))
const hasAvailableFormats = computed(() => Object.values(availableFormats).some(v => v))

// Build full URL from base - handles both absolute URLs and local paths
const buildUrl = (format) => {
  const path = `${props.base}/${props.id}.${format}`
  // If base starts with / it's a local path, use current origin
  if (props.base.startsWith('/')) {
    return path
  }
  // If base is already a full URL, use as-is
  if (props.base.startsWith('http://') || props.base.startsWith('https://')) {
    return path
  }
  // Otherwise treat as relative path
  return path
}

const formatClasses = {
  pdf: 'bg-red-100 text-red-700 hover:bg-red-200',
  epub: 'bg-green-100 text-green-700 hover:bg-green-200',
  djvu: 'bg-blue-100 text-blue-700 hover:bg-blue-200',
  mobi: 'bg-purple-100 text-purple-700 hover:bg-purple-200',
  txt: 'bg-gray-100 text-gray-700 hover:bg-gray-200'
}

onMounted(async () => {
  // Check availability of each format in parallel
  const checks = props.formats.map(async (format) => {
    checking[format] = true
    try {
      const url = buildUrl(format)
      const response = await fetch(url, { method: 'HEAD' })
      availableFormats[format] = response.ok
    } catch {
      availableFormats[format] = false
    } finally {
      checking[format] = false
    }
  })

  await Promise.all(checks)
})
</script>
