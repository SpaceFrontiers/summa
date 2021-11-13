<template lang="pug">
div
  form
    div.input-group
      input.form-control.form-control-md(v-model="query" type="text" placeholder="query")
      button.btn.btn-dark(type="submit" @click.stop.prevent="submit(true)") Search
  i.small.ms-3(v-if="total_documents !== null") {{ total_documents }} found
  .search-list.mt-4(v-if="!loading")
    search-list(:scored_documents='scored_documents')
    nav(v-if="has_next")
      ul.pagination.justify-content-center
        li.page-item(v-on:click="page -= 1;")
          a.page-link &lt;
        li.page-item.disabled
          a.page-link {{ page }}
        li.page-item(v-on:click="page += 1;")
          a.page-link &gt;
  .d-flex.justify-content-center.m-5(v-else)
    .spinner-border(role="status")
</template>

<script lang="ts">
// @ts-nocheck
import { defineComponent } from "vue";
import SearchList from "@/components/SearchList.vue";
import { useRoute } from "vue-router";
import { db } from "../database";
import { liveQuery } from "dexie";
import { useObservable } from "@vueuse/rxjs";

function add_exp_decay(
  field_name: string,
  origin: number,
  scale: number,
  offset: number,
  decay
) {
  const lamb = `log(e(), ${decay}) / ${scale}`;
  return `e() ^ ((${lamb}) * max(0, abs(${field_name} - ${origin}) - ${offset}))`;
}

function default_collectors(index_name, page) {
  const now = Date.now() / 1000;
  const defaults = {
    default: [
      {
        collector: {
          top_docs: {
            limit: page * 5,
            offset: 0,
            snippets: { description: 400, title: 140, content: 400 },
            explain: false,
            fields: [],
          },
        },
      },
      { collector: { count: {} } },
    ],
    nexus_books: [
      {
        collector: {
          top_docs: {
            limit: page * 5,
            snippets: { description: 400, title: 140 },
            scorer: {
              scorer: {
                eval_expr: `original_score * ${add_exp_decay(
                  "issued_at",
                  now - (now % 86400),
                  365.25 * 14 * 86400,
                  30 * 86400,
                  0.85
                )}`,
              },
            },
          },
        },
      },
      { collector: { count: {} } },
    ],
    nexus_media: [
      {
        collector: {
          top_docs: {
            limit: page * 5,
            snippets: { content: 400, title: 140 },
            scorer: {
              scorer: {
                eval_expr: `original_score * ${add_exp_decay(
                  "registered_at",
                  now - (now % 86400),
                  365.25 * 7 * 86400,
                  30 * 86400,
                  0.85
                )}`,
              },
            },
          },
        },
      },
      { collector: { count: {} } },
    ],
  };
  if (index_name in defaults) {
    return defaults[index_name];
  }
  return defaults["default"];
}

export default defineComponent({
  name: "SearchView",
  components: { SearchList },
  props: {
    p: {
      type: Number,
    },
    q: {
      type: String,
    },
  },
  data() {
    return {
      index_configs: useObservable(
        liveQuery(async () => {
          return db.index_configs.toArray();
        })
      ),
      loading: false,
      page: 1,
      query: "" as String,
      scored_documents: [],
      total_documents: null,
      has_next: false,
    };
  },
  async created() {
    const route = useRoute();
    if (route.query.q) {
      this.query = route.query.q.toString();
    }
    if (route.query.p) {
      this.page = parseInt(route.query.p.toString() || "1");
    }
    await this.submit(false);
  },
  watch: {
    async page() {
      if (this.page < 1) {
        this.page = 1;
      } else {
        await this.submit(false);
      }
    },
    "$route.query.q": {
      immediate: true,
      async handler(new_q) {
        if (!new_q) {
          this.query = "";
          this.has_next = false;
          await this.submit(false);
        }
      },
    },
  },
  methods: {
    async submit(new_search: Boolean) {
      if (new_search) {
        this.page = 1;
      }
      if (this.query) {
        this.loading = true;
        const enabled_index_configs = await db.index_configs
          .filter((index_config) => index_config.is_enabled)
          .toArray();
        let collector_outputs = await this.web_index_service.search(
          enabled_index_configs.map((index_config) => {
            return {
              index_name: index_config.index_payload.name,
              query: { query: { match: { value: this.query } } },
              collectors: default_collectors(
                index_config.index_payload.name,
                this.page
              ),
            };
          })
        );
        this.scored_documents =
          collector_outputs[0].collector_output.top_docs.scored_documents.slice(
            (this.page - 1) * 5,
            this.page * 5
          );
        this.total_documents =
          collector_outputs[1].collector_output.count.count;
        this.has_next = collector_outputs[0].collector_output.top_docs.has_next;
        this.loading = false;
      } else {
        this.scored_documents = [];
        this.total_documents = null;
        this.page = 1;
      }
      this.$router.push({
        name: "search",
        query: { q: this.query, p: this.page },
      });
    },
  },
});
</script>

<style scoped>
.search-list {
  padding-top: 15px;
  padding-bottom: 15px;
}
</style>
