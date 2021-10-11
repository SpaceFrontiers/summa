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
          a.page-link(href="#", tabindex="-1") &lt;
        li.page-item.disabled
          a.page-link(href="#") {{ page }}
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
import { useWebIndexStore } from "../store/web_index";

export default defineComponent({
  name: "SearchView",
  components: { SearchList },
  props: {
    i: {
      type: String,
    },
    p: {
      type: Number,
    },
    q: {
      type: String,
    },
  },
  data() {
    const web_index_store = useWebIndexStore();
    return {
      index_name: web_index_store.names.keys().next().value,
      loading: false,
      page: 1,
      query: "" as String,
      scored_documents: [],
      total_documents: null,
      has_next: false,
      web_index_store: web_index_store,
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
    if (route.query.i) {
      this.index_name = route.query.i.toString();
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
        let collector_outputs = await this.web_index_service.search(
          this.index_name,
          { query: { match: { value: this.query } } },
          [
            {
              collector: {
                top_docs: {
                  limit: 5,
                  offset: (this.page - 1) * 5,
                  snippets: { description: 200, title: 140 },
                  explain: true,
                  fields: [],
                },
              },
            },
            { collector: { count: {} } },
          ]
        );
        this.scored_documents =
          collector_outputs[0].collector_output.top_docs.scored_documents;
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
