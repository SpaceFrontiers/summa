import { createApp } from "vue";

import App from "./App.vue";
import router from "./router";
import { WebIndexService } from "./services/web-index-service";

import "./scss/styles.scss";
import * as bootstrap from "bootstrap";

const app = createApp(App);

app.use(router);


app.config.globalProperties.web_index_service = new WebIndexService();
app.mount("#app");
