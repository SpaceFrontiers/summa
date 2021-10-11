<template lang="pug">
form.mb-4.col-md-7
  div.input-group
    input.form-control.form-control-sm(v-model="new_index_ipns_path" type="text" placeholder="/ipns/...")
    button.btn.btn-dark(type="submit" @click.stop.prevent="install_new_index(new_index_ipns_path)") Install
.row.row-cols-1.row-cols-md-2.g-4(v-for="web_index in web_indices")
  .col
    .card(:class="web_index.enabled ? 'bg-light text-dark' : 'bg-warning text-white'")
      .card-body
        h5.card-title.font-monospace {{ web_index.name }}
        p.card-text {{ web_index.description }}
        .btn-group(role="group").float-end
          button.btn.btn-sm.btn-primary(@click.stop.prevent="copy_pin(web_index.ipfs_path)") Copy Pin
          button.btn.btn-sm.btn-primary(v-if="web_index.enabled" type="button" @click.stop.prevent="install_new_index(web_index.ipns_path)")
            i.bi-arrow-repeat
          button.btn.btn-sm(:class="web_index.enabled ? 'btn-danger' : 'btn-success'" type="button" @click.stop.prevent="web_index.enabled = !web_index.enabled")
            i.bi-power
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { format_bytes } from "../utils";

export default defineComponent({
  name: "DatabasesView",
  data() {
    return {
      new_index_ipns_path: "",
      web_indices: [] as {}[],
    };
  },
  async created() {
    this.web_indices = await this.web_index_service.metadatas();
  },
  methods: {
    copy_pin(ipfs_path: String) {
      navigator.clipboard.writeText("ipfs pin add " + ipfs_path);
    },
    format_bytes: format_bytes,
    async install_new_index(ipns_path: string) {
      try {
        const ipfs_path = await this.ipfs.resolve(ipns_path);
        const web_index_coordinate = await this.web_index_service.resolve(
          ipfs_path
        );
        await this.web_index_service.add_index(
          ipns_path,
          ipfs_path,
          web_index_coordinate
        );
      } catch (e) {
        console.error(e);
      }
    },
  },
});
</script>
