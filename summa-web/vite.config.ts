import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import vue from "@vitejs/plugin-vue";
import vuePugPlugin from "vue-pug-plugin";
import wasm from "vite-plugin-wasm";

// https://vitejs.dev/config/
export default defineConfig({
  base: "",
  build: {
    assetsInlineLimit: 0,
    rollupOptions: {
      input: {
        index: "./index.html",
        "service-worker": "./service-worker.js",
        localforage: "./node_modules/localforage/dist/localforage.js",
      },
      output: {
        entryFileNames: (asset_info) => {
          if (asset_info.name === "service-worker") {
            return "[name].js";
          }
          return "assets/[name].[hash].js";
        },
      },
    },
    target: "modules",
  },
  plugins: [
    vue({
      template: {
        preprocessOptions: {
          // 'preprocessOptions' is passed through to the pug compiler
          plugins: [vuePugPlugin],
        },
      },
    }),
    wasm(),
    topLevelAwait(),
  ],
  optimizeDeps: {
    esbuildOptions: {
      target: "es2022",
    },
    include: [
      "@libp2p/logger",
      "@multiformats/multiaddr",
      "ipfs-core-types",
      "ipfs-http-client",
      "localforage",
      "merge-options",
      "summa-wasm",
    ],
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      "~": fileURLToPath(new URL("./node_modules", import.meta.url)),
    },
    preserveSymlinks: true,
  },
  server: {
    fs: {
      // Allow serving files from one level up to the project root
      allow: [".."],
    },
  },
});
