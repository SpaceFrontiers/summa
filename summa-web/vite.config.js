import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import wasm from 'vite-plugin-wasm'

export default defineConfig({
  base: './',
  plugins: [
    vue(),
    wasm()
  ],
  // esnext keeps native top-level await for the wasm-bindgen module.
  build: {
    target: 'esnext',
  },
  optimizeDeps: {
    exclude: ['summa-wasm']
  }
})
