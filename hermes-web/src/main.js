import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import { useAppConfig } from './composables/useAppConfig'
// Import useHermes to trigger WASM preload (singleton promise starts on import)
import './composables/useHermes'

// Detect if app is served from IPFS gateway on startup
const appConfig = useAppConfig()
appConfig.detect()

const app = createApp(App)
app.use(createPinia())
app.mount('#app')
