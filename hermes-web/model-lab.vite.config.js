import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('.', import.meta.url))

export default {
  base: './',
  // The lab uses self-contained CSS and intentionally does not require the
  // search application's Tailwind/PostCSS toolchain (or its WASM package).
  css: {
    postcss: { plugins: [] },
  },
  build: {
    outDir: 'dist-model-lab',
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(root, 'model-lab.html'),
    },
  },
}
