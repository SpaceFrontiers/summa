import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url))

export default {
  base: './',
  // Model Lab owns its CSS and has no dependency on the search app's
  // Tailwind/PostCSS or WASM pipeline.
  css: {
    postcss: { plugins: [] }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      // Retain the URL used by the live hermes-llm server and static bundles.
      input: resolve(root, 'model-lab.html')
    }
  }
}
