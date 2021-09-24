<script lang="ts">
import {defineComponent} from "vue";
import {RouterLink, RouterView} from "vue-router";
import {format_bytes} from "./utils";
import {useStatsStore} from "./store/stats";
import {useIpfsStore} from "./store/ipfs";

export default defineComponent({
  name: "App",
  components: {RouterLink, RouterView},
  data() {
    let ipfs_store = useIpfsStore();
    let stats_store = useStatsStore();
    return { ipfs_store, stats_store, is_alive: true }
  },
  methods: {
    format_bytes: format_bytes,
  },
  async created() {
    try {
      await this.ipfs.id();
    } catch (e) {
      this.is_alive = false
      return
    }
    // @ts-ignore
    let web_index = await this.ipfs_store.setup(
        "page",
        "k51qzi5uqu5dg9kaxpau2an7ae09yl4yrgna5x8uunvwn5wy27tr7r8uddxn72",
        ["text", "title"],
        ["category"]
    )
    await this.summa.setup(web_index);
    this.stats_store.set_stats(await this.summa.stats());
  },
});
</script>

<template lang="pug">
div.d-flex.flex-column.min-vh-100.w-100
  header
    nav.navbar.navbar-expand-lg.navbar-dark.bg-dark
      .container
        a.navbar-brand Summa Web
        ul.navbar-nav.me-auto.mb-2.mb-lg-0(v-if="is_alive")
          li.nav-item
            router-link.nav-link(to="/") Search
          li.nav-item
            router-link.nav-link(to="/manage") Manage
        span.navbar-text.font-monospace #{format_bytes(stats_store.stats.downloaded_bytes)} / #{stats_store.stats.requests}R
  .container.col-md-7.offset-md-1.mt-5
    RouterView(v-if="is_alive")
    div(v-else).small
      .alert.alert-danger(role="alert")
        b Local IPFS Daemon is not launched
      p Summa Web requires installed and configured IPFS Desktop
      ul
        li
          a(href="https://docs.ipfs.tech/install/ipfs-desktop/") Install IPFS
        li Configure CORS Headers in Terminal
          .container
            code
              | ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
            br
            code
              | ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["GET", "POST"]'
        li Install IPFS Companion (optionally)
          ul
            li
              a(href="https://chrome.google.com/webstore/detail/ipfs-companion/nibjojkomfdiaoajekhjakgkdhaomnch") Chrome
            li
              a(href="https://addons.mozilla.org/en-US/firefox/addon/ipfs-companion/") Firefox
        li
          a(href=".") Refresh this page
  footer.footer.mt-auto.bg-dark
    .container.clearfix
      .float-end.small
        a.link-light(href="https://github.com/izihawa/summa") Summa Powered 2022
</template>
