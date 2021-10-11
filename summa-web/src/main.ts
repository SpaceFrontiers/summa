import { createApp } from "vue";
import { createPinia } from "pinia";

import App from "./App.vue";
import router from "./router";

import "./scss/styles.scss";
import * as bootstrap from "bootstrap";

const app = createApp(App);
const pinia = createPinia();

app.use(pinia).use(router);

import { ipfs } from "./plugins/ipfs";
import { WebIndexService } from "./plugins/web-index-service";

app.config.globalProperties.ipfs = ipfs;
app.config.globalProperties.web_index_service = new WebIndexService();
app.mount("#app");
