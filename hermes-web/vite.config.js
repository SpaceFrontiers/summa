import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  base: './',
  plugins: [
    vue(),
    wasm(),
    topLevelAwait()
  ],
  build: {
    target: 'esnext',
  },
  optimizeDeps: {
    exclude: ['hermes-wasm']
  }
})
