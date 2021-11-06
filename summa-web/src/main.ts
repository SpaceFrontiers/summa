import { createApp } from "vue";

import App from "./App.vue";
import router from "./router";

import "./scss/styles.scss";
import * as bootstrap from "bootstrap";

const app = createApp(App);

app.use(router);

import { ipfs } from "./services/ipfs";
import { WebIndexService } from "./services/web-index-service";

app.config.globalProperties.ipfs = ipfs;
app.config.globalProperties.web_index_service = new WebIndexService();
app.mount("#app");
