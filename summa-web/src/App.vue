<template lang="pug">
div.d-flex.flex-column.min-vh-100.w-100
  header
    nav.navbar.navbar-expand-sm.navbar-dark.bg-dark
      .container-fluid
        a.navbar-brand Summa
        button.navbar-toggler(type="button" data-bs-toggle="collapse" data-bs-target="#navbar-nav" aria-controls="navbar-nav" aria-expanded="false" aria-label="Toggle navigation")
          span.navbar-toggler-icon
        .collapse.navbar-collapse(id="navbar-nav")
          .navbar-nav
            router-link.nav-link(to="/") Search
            router-link.nav-link(to="/databases") Databases
            router-link.nav-link(to="/about") About
        span.navbar-text.font-monospace #{format_bytes(cache_metrics.requests_bytes)}
  .mt-5
    .container(v-if="is_loading")
      is-loading-view(:label="loading_label")
    .container(v-else-if="is_loading_failed")
      connectivity-issues-view
    .container.col-md-7.offset-md-1(v-else)
      router-view
  footer.footer.mt-auto.bg-dark
    .container.clearfix
      .float-end.small
        a.link-light(href="https://github.com/izihawa/summa") Summa Powered 2022
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { RouterLink, RouterView } from "vue-router";
import IsLoadingView from "@/components/IsLoading.vue";
import { format_bytes } from "./utils";
import ConnectivityIssuesView from "./components/ConnectivityIssues.vue";
import { cache_metrics } from "./plugins/web-index-service";
import * as localforage from "localforage";

export default defineComponent({
  name: "App",
  components: { ConnectivityIssuesView, RouterLink, RouterView, IsLoadingView },
  data() {
    return {
      is_loading: true,
      is_loading_failed: false,
      loading_label: "",
      cache_metrics: cache_metrics,
    };
  },
  methods: {
    format_bytes: format_bytes,
    status_callback(type: string, message: string) {
      if (type == "status") {
        this.loading_label = message;
      }
    },
  },

  async created() {
    this.is_loading = true;
    this.web_index_service.status_callback = this.status_callback;
    const is_succeed = await this.web_index_service.setup({ num_threads: 8 });
    this.is_loading_failed = !is_succeed;
    this.is_loading = false;
  },
});
</script>
