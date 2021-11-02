<template lang="pug">
form.mb-4.col-md-7
  div.input-group
    input.form-control.form-control-sm(v-model="new_index_ipns_path" type="text" placeholder="/ipns/...")
    button.btn.btn-dark(type="submit" @click.stop.prevent="install_new_index(new_index_ipns_path)") Install
.row.row-cols-1.row-cols-md-2.g-4
  .col(v-for="[index_name, index_config] of index_configs")
    .card(:class="index_config.enabled ? 'bg-light text-dark' : 'bg-warning text-white'")
      .card-body
        h5.card-title.font-monospace {{ index_name }}
        p.card-text {{ index_payloads.get(index_name) ? index_payloads.get(index_name).description : "" }}
        .btn-group(role="group").float-end
          button.btn.btn-sm(:class="index_config.enabled ? 'btn-danger' : 'btn-success'" type="button" @click.stop.prevent="web_index_store.switch(index_name)")
            i.bi-power
          button.btn.btn-danger.btn-sm(type="button" @click.stop.prevent="web_index_service.delete_index(index_name)")
            i.bi-trash
  .col(v-if="loading")
    .card.bg-light.text-dark
      .card-body
        .d-flex.justify-content-center
          .spinner-border(role="status")
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { format_bytes } from "../utils";
import { useWebIndexStore } from "../store/web_index";

export default defineComponent({
  name: "DatabasesView",
  async created() {
    this.index_payloads = await this.web_index_service.get_index_payloads();
  },
  data() {
    const web_index_store = useWebIndexStore();
    return {
      index_configs: web_index_store.index_configs,
      index_payloads: new Map(),
      loading: false,
      new_index_ipns_path: "",
      web_index_store: web_index_store,
    };
  },
  methods: {
    copy_pin(ipfs_path: String) {
      navigator.clipboard.writeText("ipfs pin add " + ipfs_path);
    },
    format_bytes: format_bytes,
    async get_index_payload(index_name: String) {
      return await this.web_index_service.get_index_payload(index_name);
    },
    async install_new_index(ipns_path: String) {
      this.loading = true;
      try {
        await this.web_index_service.install_index(ipns_path);
      } finally {
        this.loading = false;
      }
    },
  },
});
</script>
