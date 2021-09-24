<template lang="pug">
div(v-for="web_index in ipfs_store.web_indices")
  .btn-group.float-end
    button.btn.btn-sm.btn-dark(type="button" @click.stop.prevent="pin(web_index.name)")
      i.bi-cloud-download-fill
    button.btn.btn-sm.btn-dark(type="button" @click.stop.prevent="update(web_index.name)")
      i.bi-bootstrap-reboot
  h4.font-monospace #{web_index.name}
  p.font-monospace Cached: #{format_bytes(web_index.local_size)} of #{format_bytes(web_index.size)} (#{format_percent(web_index.local_size / web_index.size)})
  p.font-monospace.mt-3 IPNS: #{web_index.name_hash}
  p.font-monospace IPFS: #{web_index.path_hash}
  hr
</template>

<script lang="ts">
import {defineComponent} from "vue";
import { useIpfsStore } from "@/store/ipfs";
import {format_bytes, format_percent} from "../utils";

export default defineComponent({
  name: "ManageView",
  data() {
    return {
      ipfs_store: useIpfsStore(),
    }
  },
  async mounted() {
    for (let web_index of this.ipfs_store.web_indices) {
      // @ts-ignore
      await this.ipfs_store.update_size(web_index.name)
    }
  },
  methods: {
    async update(name: string) {
      // @ts-ignore
      await this.ipfs_store.update(name)
    },
    async pin(name: string) {
      // @ts-ignore
      let web_index = this.ipfs_store.lookup(name)
      await this.ipfs.pin.add(web_index.path_hash)
    },
    format_percent: format_percent,
    format_bytes: format_bytes,
  }
})

</script>
