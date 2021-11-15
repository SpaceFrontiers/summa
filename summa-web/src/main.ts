import { createApp } from "vue";

import App from "./App.vue";
import router from "./router";
import { SearchService } from "./services/web-index-service";

import "./scss/styles.scss";
import * as bootstrap from "bootstrap";

const app = createApp(App);

app.use(router);

app.config.globalProperties.search_service = new SearchService();
app.mount("#app");
