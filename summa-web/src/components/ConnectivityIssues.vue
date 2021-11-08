<template lang="pug">
div.font-monospace(v-if="is_localhost")
  p Is local IPFS daemon enabled?
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { ipfs_url } from "@/options";

export default defineComponent({
  name: "ConnectivityIssues",
  computed: {
    is_localhost() {
      return ipfs_url.includes("localhost");
    },
  },
});
</script>
