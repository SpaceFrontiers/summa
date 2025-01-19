import { ref, computed } from 'vue'
import { parseUxConfig, DEFAULT_CONFIG, interpolateUrl, formatFieldValue, getFieldClasses, parseUris, splitContent, languageToEmoji, formatUnixYear } from '../lib/uxConfigParser'

// Singleton state
const config = ref({ ...DEFAULT_CONFIG })
const isLoaded = ref(false)
const error = ref(null)

/**
 * Composable for managing UX configuration
 *
 * Loads ux.dsl from the index directory and provides
 * configuration for rendering search results.
 */
export function useUxConfig() {

  /**
   * Load UX config from a fetch function
   *
   * @param {Function} fetchFn - Function to fetch file content: (path) => Promise<Uint8Array>
   * @param {string} basePath - Base path of the index (e.g., /ipfs/<CID>)
   */
  const load = async (fetchFn, basePath) => {
    error.value = null

    try {
      const uxPath = basePath.endsWith('/') ? `${basePath}ux.dsl` : `${basePath}/ux.dsl`
      console.log('Loading UX config from:', uxPath)

      const data = await fetchFn(uxPath)
      const text = new TextDecoder().decode(data)

      config.value = parseUxConfig(text)
      isLoaded.value = true
      console.log('UX config loaded:', config.value)
    } catch (e) {
      // ux.dsl is optional, use defaults if not found
      console.log('No ux.dsl found, using defaults')
      config.value = { ...DEFAULT_CONFIG }
      isLoaded.value = true
    }
  }

  /**
   * Reset to default config
   */
  const reset = () => {
    config.value = { ...DEFAULT_CONFIG }
    isLoaded.value = false
    error.value = null
  }

  /**
   * Get display columns (from config or all fields)
   *
   * @param {string[]} allFields - All available fields from schema
   * @returns {string[]} Fields to display
   */
  const getDisplayColumns = (allFields) => {
    if (config.value.layout.columns && config.value.layout.columns.length > 0) {
      return config.value.layout.columns
    }
    return allFields
  }

  /**
   * Get field configuration
   *
   * @param {string} fieldName - Field name
   * @returns {object} Field config
   */
  const getFieldConfig = (fieldName) => {
    return config.value.layout.fields[fieldName] || {}
  }

  /**
   * Get field label
   *
   * @param {string} fieldName - Field name
   * @returns {string|null} Display label or null if not specified in config
   */
  const getFieldLabel = (fieldName) => {
    const fieldConfig = getFieldConfig(fieldName)
    // Only show label if explicitly set in config
    return fieldConfig.label || null
  }

  /**
   * Format a field value for display
   *
   * @param {string} fieldName - Field name
   * @param {any} value - Raw value
   * @returns {string|object} Formatted value or complex object for special formats
   */
  const formatField = (fieldName, value) => {
    const fieldConfig = getFieldConfig(fieldName)

    // Handle special formats that return complex objects
    switch (fieldConfig.format) {
      case 'uri_links':
        return { type: 'uri_links', links: parseUris(value) }
      case 'split_newline':
        return { type: 'split_newline', lines: splitContent(value, '\n') }
      case 'language_emoji':
        return languageToEmoji(value)
      case 'unix_year':
        return formatUnixYear(value)
      case 'ipfs_files':
        return {
          type: 'ipfs_files',
          id: value,
          formats: fieldConfig.ipfs_formats || ['pdf', 'epub', 'djvu'],
          base: fieldConfig.ipfs_base || ''
        }
      default:
        return formatFieldValue(value, fieldConfig)
    }
  }

  /**
   * Check if a field has a special render type
   */
  const getFieldRenderType = (fieldName) => {
    const fieldConfig = getFieldConfig(fieldName)
    const format = fieldConfig.format
    if (['uri_links', 'split_newline', 'ipfs_files'].includes(format)) {
      return format
    }
    return 'text'
  }

  /**
   * Get CSS classes for a field
   *
   * @param {string} fieldName - Field name
   * @returns {string} CSS classes
   */
  const getFieldStyles = (fieldName) => {
    const fieldConfig = getFieldConfig(fieldName)
    return getFieldClasses(fieldConfig)
  }

  /**
   * Get field width style
   *
   * @param {string} fieldName - Field name
   * @returns {object} Style object
   */
  const getFieldWidth = (fieldName) => {
    const fieldConfig = getFieldConfig(fieldName)
    if (fieldConfig.width) {
      return { width: fieldConfig.width }
    }
    return {}
  }

  /**
   * Check if a field is clickable
   *
   * @param {string} fieldName - Field name
   * @returns {boolean}
   */
  const isFieldClickable = (fieldName) => {
    return !!config.value.actions[fieldName]
  }

  /**
   * Get click action for a field
   *
   * @param {string} fieldName - Field name
   * @param {object} doc - Document data
   * @returns {object|null} Action config with interpolated URL
   */
  const getFieldAction = (fieldName, doc) => {
    const action = config.value.actions[fieldName]
    if (!action) return null

    return {
      ...action,
      url: action.url ? interpolateUrl(action.url, doc) : null,
      value: action.value ? interpolateUrl(action.value, doc) : null
    }
  }

  /**
   * Handle field click
   *
   * @param {string} fieldName - Field name
   * @param {object} doc - Document data
   */
  const handleFieldClick = (fieldName, doc) => {
    const action = getFieldAction(fieldName, doc)
    if (!action) return

    switch (action.action) {
      case 'copy':
        if (action.value) {
          navigator.clipboard.writeText(decodeURIComponent(action.value))
          console.log('Copied to clipboard:', action.value)
        }
        break
      case 'custom':
        // Emit custom event for handling elsewhere
        break
      case 'navigate':
      default:
        if (action.url) {
          if (action.target === '_blank') {
            window.open(action.url, '_blank')
          } else {
            window.location.href = action.url
          }
        }
    }
  }

  /**
   * Check if row click is enabled
   *
   * @returns {boolean}
   */
  const hasRowClick = computed(() => {
    return !!config.value.rowClick
  })

  /**
   * Handle row click
   *
   * @param {object} doc - Document data
   */
  const handleRowClick = (doc) => {
    if (!config.value.rowClick) return

    const url = interpolateUrl(config.value.rowClick.url, doc)
    if (url) {
      if (config.value.rowClick.target === '_blank') {
        window.open(url, '_blank')
      } else {
        window.location.href = url
      }
    }
  }

  return {
    // State
    config,
    isLoaded,
    error,

    // Computed
    title: computed(() => config.value.title),
    short_name: computed(() => config.value.short_name),
    description: computed(() => config.value.description),
    placeholder: computed(() => config.value.placeholder),
    logo: computed(() => config.value.logo),
    hasRowClick,

    // Methods
    load,
    reset,
    getDisplayColumns,
    getFieldConfig,
    getFieldLabel,
    formatField,
    getFieldRenderType,
    getFieldStyles,
    getFieldWidth,
    isFieldClickable,
    getFieldAction,
    handleFieldClick,
    handleRowClick
  }
}
