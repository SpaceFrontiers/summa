<template lang="pug">
form.mb-4.col-md-7
  div.input-group
    input.form-control.form-control-sm(v-model="new_index_ipns_path" type="text" placeholder="/ipns/...")
    button.btn.btn-dark(type="submit" @click.stop.prevent="install_new_index(new_index_ipns_path)") Install
.row.row-cols-1.row-cols-md-2.g-4
  .col(v-for="index_config of index_configs")
    .card(:class="index_config.is_enabled ? 'bg-light text-dark' : 'bg-warning text-white'")
      .card-body
        h5.card-title.font-monospace {{ index_config.index_payload.name }}
        p.card-text {{ index_config.index_payload.description  }}
        .card-body.small
          .row
            hr.mt-4
            .form-check.col-4.mt-4
              input.form-check-input(type="checkbox" :id="'checkbox_warmup_' + index_config.index_payload.name" v-model="index_config.is_warm_up" @change="switch_warmup(index_config)")
              label.form-check-label(:for="'checkbox_warmup_' + index_config.index_payload.name") Warm Up
            .form-text.col-8.mt-3
              span.lh-1 Cache most important index parts in browser for better performance
          .row
            .form-check.col-4.mt-4
              input.form-check-input(type="checkbox" :id="'checkbox_enabled_' + index_config.index_payload.name" v-model="index_config.is_enabled" @change="index_config.save()")
              label.form-check-label(:for="'checkbox_enabled_' + index_config.index_payload.name") Enabled
          .row
            hr.mt-4
            span Index ID:
            .form-text
              span.lh-1 {{ index_config.index_seed }}
          .btn-group(role="group").float-end
            button.btn.btn-danger.btn-sm(type="button" @click.stop.prevent="search_service.delete_index(index_config.index_payload.name)")
              i.bi-trash
  .col(v-if="is_loading")
    .card.bg-light.text-dark
      .card-body
        is-loading-view(label="installing...")
</template>

<script lang="ts">
import { liveQuery } from "dexie";
import { defineComponent } from "vue";
import { format_bytes } from "@/utils";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import { db, IndexConfig } from "@/database";
import { useObservable } from "@vueuse/rxjs";
import IsLoadingView from "@/components/IsLoading.vue";
import { IpfsDatabaseSeed } from "../services/search-service";

export default defineComponent({
  name: "DatabasesView",
  components: { IsLoadingView },
  data() {
    return {
      index_configs: useObservable(
        liveQuery(async () => {
          return db.index_configs.toArray();
        })
      ),
      is_loading: false,
      new_index_ipns_path: "",
    };
  },
  methods: {
    format_bytes: format_bytes,
    async install_new_index(ipfs_path: string) {
      this.is_loading = true;
      try {
        await this.search_service.add_index({
          seed: new IpfsDatabaseSeed(ipfs_path),
          is_enabled: true,
        });
      } finally {
        this.is_loading = false;
      }
    },
    async switch_warmup(index_config: IndexConfig) {
      if (index_config.is_warm_up) {
        await this.search_service.web_index_service_worker.warmup(
          index_config.index_payload.name
        );
      }
      await index_config.save();
    },
  },
});
</script>
