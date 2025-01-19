/**
 * Parser for Hermes Web UX Configuration DSL
 *
 * Parses ux.dsl files that configure how search results are rendered.
 */

/**
 * Default UX configuration
 */
export const DEFAULT_CONFIG = {
  title: 'Hermes Search',
  short_name: null,  // Short name for display in connection lists
  description: '',
  placeholder: 'Search...',
  logo: null,
  layout: {
    columns: [],
    fields: {}
  },
  actions: {},
  rowClick: null,
  styles: {}
}

/**
 * Tokenize the DSL input
 */
function tokenize(input) {
  const tokens = []
  let i = 0

  while (i < input.length) {
    // Skip whitespace
    if (/\s/.test(input[i])) {
      i++
      continue
    }

    // Skip comments
    if (input[i] === '#') {
      while (i < input.length && input[i] !== '\n') i++
      continue
    }

    // String literals
    if (input[i] === '"' || input[i] === "'") {
      const quote = input[i]
      i++
      let str = ''
      while (i < input.length && input[i] !== quote) {
        if (input[i] === '\\' && i + 1 < input.length) {
          i++
          str += input[i]
        } else {
          str += input[i]
        }
        i++
      }
      i++ // skip closing quote
      tokens.push({ type: 'string', value: str })
      continue
    }

    // Brackets and braces
    if ('{[()]}'.includes(input[i])) {
      tokens.push({ type: input[i], value: input[i] })
      i++
      continue
    }

    // Comma
    if (input[i] === ',') {
      tokens.push({ type: ',', value: ',' })
      i++
      continue
    }

    // Identifiers and keywords
    if (/[a-zA-Z_]/.test(input[i])) {
      let id = ''
      while (i < input.length && /[a-zA-Z0-9_]/.test(input[i])) {
        id += input[i]
        i++
      }
      tokens.push({ type: 'identifier', value: id })
      continue
    }

    // Numbers
    if (/[0-9]/.test(input[i])) {
      let num = ''
      while (i < input.length && /[0-9.%]/.test(input[i])) {
        num += input[i]
        i++
      }
      tokens.push({ type: 'number', value: num })
      continue
    }

    // Skip unknown characters
    i++
  }

  return tokens
}

/**
 * Parse tokens into config object
 */
function parse(tokens) {
  const config = { ...DEFAULT_CONFIG, layout: { columns: [], fields: {} }, actions: {}, styles: {} }
  let i = 0

  function peek() {
    return tokens[i]
  }

  function consume(expectedType) {
    const token = tokens[i]
    if (expectedType && token?.type !== expectedType) {
      throw new Error(`Expected ${expectedType}, got ${token?.type}`)
    }
    i++
    return token
  }

  function parseArray() {
    consume('[')
    const items = []
    while (peek()?.type !== ']') {
      const token = consume()
      if (token.type === 'identifier' || token.type === 'string') {
        items.push(token.value)
      }
      if (peek()?.type === ',') consume(',')
    }
    consume(']')
    return items
  }

  function parseBlock() {
    consume('{')
    const block = {}
    while (peek()?.type !== '}') {
      const key = consume('identifier').value

      if (peek()?.type === '{') {
        block[key] = parseBlock()
      } else if (peek()?.type === '[') {
        block[key] = parseArray()
      } else if (peek()?.type === 'string') {
        block[key] = consume('string').value
      } else if (peek()?.type === 'number') {
        block[key] = consume('number').value
      } else if (peek()?.type === 'identifier') {
        block[key] = consume('identifier').value
      }
    }
    consume('}')
    return block
  }

  function _parseFieldBlock() {
    const fieldName = consume('identifier').value
    const fieldConfig = parseBlock()
    return { name: fieldName, config: fieldConfig }
  }

  // Main parsing loop
  while (i < tokens.length) {
    const token = peek()
    if (!token) break

    if (token.type === 'identifier') {
      const keyword = consume('identifier').value

      switch (keyword) {
        case 'title':
          config.title = consume('string').value
          break

        case 'short_name':
          config.short_name = consume('string').value
          break

        case 'description':
          config.description = consume('string').value
          break

        case 'placeholder':
          config.placeholder = consume('string').value
          break

        case 'logo':
          config.logo = consume('string').value
          break

        case 'layout': {
          const layoutBlock = parseBlock()
          if (layoutBlock.columns) {
            config.layout.columns = layoutBlock.columns
          }
          // Parse field definitions
          for (const [key, value] of Object.entries(layoutBlock)) {
            if (key !== 'columns' && typeof value === 'object') {
              config.layout.fields[key] = value
            }
          }
          break
        }

        case 'actions': {
          const actionsBlock = parseBlock()
          for (const [key, value] of Object.entries(actionsBlock)) {
            if (key === 'click' || key.startsWith('click_')) {
              // Handle nested click blocks
              config.actions[key] = value
            } else {
              config.actions[key] = value
            }
          }
          break
        }

        case 'row_click':
          config.rowClick = parseBlock()
          break

        case 'styles':
          config.styles = parseBlock()
          break

        default:
          // Skip unknown keywords
          if (peek()?.type === '{') {
            parseBlock()
          } else if (peek()?.type === '[') {
            parseArray()
          } else if (peek()?.type === 'string') {
            consume('string')
          }
      }
    } else {
      i++
    }
  }

  return config
}

/**
 * Parse a UX DSL string into a configuration object
 *
 * @param {string} dsl - The DSL content
 * @returns {object} Parsed configuration
 */
export function parseUxConfig(dsl) {
  try {
    const tokens = tokenize(dsl)
    return parse(tokens)
  } catch (e) {
    console.error('Failed to parse UX config:', e)
    return { ...DEFAULT_CONFIG }
  }
}

/**
 * Interpolate URL template with field values
 *
 * @param {string} template - URL template with {field} placeholders
 * @param {object} doc - Document with field values
 * @returns {string} Interpolated URL
 */
export function interpolateUrl(template, doc) {
  if (!template) return null
  return template.replace(/\{([^}]+)\}/g, (match, field) => {
    const value = doc[field]
    if (value === undefined || value === null) return ''
    return encodeURIComponent(String(value))
  })
}

/**
 * Language code to emoji mapping
 */
const LANGUAGE_EMOJI = {
  en: '🇬🇧', eng: '🇬🇧', english: '🇬🇧',
  de: '🇩🇪', deu: '🇩🇪', german: '🇩🇪',
  fr: '🇫🇷', fra: '🇫🇷', french: '🇫🇷',
  es: '🇪🇸', spa: '🇪🇸', spanish: '🇪🇸',
  it: '🇮🇹', ita: '🇮🇹', italian: '🇮🇹',
  pt: '🇵🇹', por: '🇵🇹', portuguese: '🇵🇹',
  ru: '🇷🇺', rus: '🇷🇺', russian: '🇷🇺',
  zh: '🇨🇳', zho: '🇨🇳', chinese: '🇨🇳',
  ja: '🇯🇵', jpn: '🇯🇵', japanese: '🇯🇵',
  ko: '🇰🇷', kor: '🇰🇷', korean: '🇰🇷',
  ar: '🇸🇦', ara: '🇸🇦', arabic: '🇸🇦',
  nl: '🇳🇱', nld: '🇳🇱', dutch: '🇳🇱',
  pl: '🇵🇱', pol: '🇵🇱', polish: '🇵🇱',
  uk: '🇺🇦', ukr: '🇺🇦', ukrainian: '🇺🇦',
  cs: '🇨🇿', ces: '🇨🇿', czech: '🇨🇿',
  sv: '🇸🇪', swe: '🇸🇪', swedish: '🇸🇪',
  da: '🇩🇰', dan: '🇩🇰', danish: '🇩🇰',
  fi: '🇫🇮', fin: '🇫🇮', finnish: '🇫🇮',
  no: '🇳🇴', nor: '🇳🇴', norwegian: '🇳🇴',
  he: '🇮🇱', heb: '🇮🇱', hebrew: '🇮🇱',
  tr: '🇹🇷', tur: '🇹🇷', turkish: '🇹🇷',
  el: '🇬🇷', ell: '🇬🇷', greek: '🇬🇷',
  hu: '🇭🇺', hun: '🇭🇺', hungarian: '🇭🇺',
  ro: '🇷🇴', ron: '🇷🇴', romanian: '🇷🇴',
  th: '🇹🇭', tha: '🇹🇭', thai: '🇹🇭',
  vi: '🇻🇳', vie: '🇻🇳', vietnamese: '🇻🇳',
  id: '🇮🇩', ind: '🇮🇩', indonesian: '🇮🇩',
  hi: '🇮🇳', hin: '🇮🇳', hindi: '🇮🇳',
  la: '🏛️', lat: '🏛️', latin: '🏛️',
}

/**
 * Convert language code to emoji
 */
export function languageToEmoji(lang) {
  if (!lang) return ''
  const normalized = String(lang).toLowerCase().trim()
  return LANGUAGE_EMOJI[normalized] || lang
}

/**
 * Format unix timestamp to year (with date if not Jan 1)
 */
export function formatUnixYear(timestamp) {
  if (!timestamp) return ''
  const date = new Date(Number(timestamp) * 1000)
  const year = date.getFullYear()
  const month = date.getMonth()
  const day = date.getDate()

  // If Jan 1, just show year
  if (month === 0 && day === 1) {
    return String(year)
  }

  // Otherwise show full date
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

/**
 * Split content by newlines and return as array
 */
export function splitContent(value, separator = '\n') {
  if (!value) return []
  return String(value).split(separator).filter(s => s.trim())
}

/**
 * Parse URI and extract schema-specific link
 */
export function parseUri(uri) {
  if (!uri) return null
  const match = uri.match(/^([a-z]+):\/\/(.+)$/i)
  if (!match) return null

  const [, schema, path] = match

  switch (schema.toLowerCase()) {
    case 'doi':
      return { schema: 'doi', path, url: `https://doi.org/${path}`, label: `DOI: ${path}` }
    case 'pubmed':
      return { schema: 'pubmed', path, url: `https://pubmed.ncbi.nlm.nih.gov/${path}`, label: `PubMed: ${path}` }
    case 'arxiv':
      return { schema: 'arxiv', path, url: `https://arxiv.org/abs/${path}`, label: `arXiv: ${path}` }
    case 'isbn':
      return { schema: 'isbn', path, url: `https://openlibrary.org/isbn/${path}`, label: `ISBN: ${path}` }
    case 'http':
    case 'https':
      return { schema, path, url: uri, label: uri }
    default:
      return { schema, path, url: null, label: uri }
  }
}

/**
 * Parse multiple URIs (comma-separated)
 */
export function parseUris(uris) {
  if (!uris) return []
  return String(uris).split(',').map(u => parseUri(u.trim())).filter(Boolean)
}

/**
 * Format a field value according to config
 *
 * @param {any} value - The raw value
 * @param {object} fieldConfig - Field configuration
 * @returns {string} Formatted value
 */
export function formatFieldValue(value, fieldConfig = {}) {
  if (value === undefined || value === null) {
    return fieldConfig.default || ''
  }

  let formatted = String(value)

  // Apply format
  switch (fieldConfig.format) {
    case 'number':
      formatted = Number(value).toLocaleString()
      break
    case 'currency':
      formatted = Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      break
    case 'date':
      formatted = new Date(value).toLocaleDateString()
      break
    case 'unix_year':
      formatted = formatUnixYear(value)
      break
    case 'language_emoji':
      formatted = languageToEmoji(value)
      break
    case 'bytes':
      formatted = formatBytes(Number(value))
      break
  }

  // Apply prefix/suffix
  if (fieldConfig.prefix) formatted = fieldConfig.prefix + formatted
  if (fieldConfig.suffix) formatted = formatted + fieldConfig.suffix

  // Apply truncation
  if (fieldConfig.truncate && formatted.length > fieldConfig.truncate) {
    formatted = formatted.slice(0, fieldConfig.truncate) + '...'
  }

  return formatted
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

/**
 * Get CSS classes for a field based on config
 *
 * @param {object} fieldConfig - Field configuration
 * @returns {string} CSS classes
 */
export function getFieldClasses(fieldConfig = {}) {
  const classes = []

  switch (fieldConfig.style) {
    case 'bold':
      classes.push('font-semibold')
      break
    case 'italic':
      classes.push('italic')
      break
    case 'mono':
      classes.push('font-mono text-sm')
      break
    case 'tag':
      classes.push('inline-block px-2 py-0.5 rounded-full text-xs')
      break
    case 'link':
      classes.push('text-blue-600 hover:underline cursor-pointer')
      break
  }

  if (fieldConfig.color) {
    switch (fieldConfig.color) {
      case 'red':
        classes.push(fieldConfig.style === 'tag' ? 'bg-red-100 text-red-700' : 'text-red-600')
        break
      case 'blue':
        classes.push(fieldConfig.style === 'tag' ? 'bg-blue-100 text-blue-700' : 'text-blue-600')
        break
      case 'green':
        classes.push(fieldConfig.style === 'tag' ? 'bg-green-100 text-green-700' : 'text-green-600')
        break
      case 'yellow':
        classes.push(fieldConfig.style === 'tag' ? 'bg-yellow-100 text-yellow-700' : 'text-yellow-600')
        break
      case 'purple':
        classes.push(fieldConfig.style === 'tag' ? 'bg-purple-100 text-purple-700' : 'text-purple-600')
        break
      case 'gray':
        classes.push(fieldConfig.style === 'tag' ? 'bg-gray-100 text-gray-700' : 'text-gray-600')
        break
    }
  }

  return classes.join(' ')
}
