import { createApp } from "vue";
import { createPinia } from "pinia";
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import { ipfs } from "./plugins/ipfs";
import { summa } from "./plugins/summa";

import App from "./App.vue";
import router from "./router";

import "./scss/styles.scss";

const app = createApp(App);
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate)

app.use(pinia).use(router);
app.config.globalProperties.summa = summa;
app.config.globalProperties.ipfs = ipfs;
app.mount("#app")
