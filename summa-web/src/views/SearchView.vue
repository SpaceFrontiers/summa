<script lang="ts">
// @ts-nocheck
import { defineComponent } from "vue";
import SearchList from "@/components/SearchList.vue";
import { useStatsStore } from "@/store/stats";
import { useRoute } from "vue-router";

export default defineComponent({
  name: "SearchView",
  components: { SearchList },
  props: {
    p: {
      type: Number,
    },
    q: {
      type: String
    }
  },
  data() {
    return {
      loading: false,
      page: 1,
      query: '' as String,
      scored_documents: [],
      stats_store: useStatsStore(),
      total_documents: 0,
    };
  },
  async created() {
    const route = useRoute();
    if (route.query.q) {
      this.query = route.query.q.toString();
    }
    if (route.query.p) {
      this.page = parseInt(route.query.p.toString() || '1');
    }
    await this.submit(false)
  },
  watch: {
    async page() {
      if (this.page < 1) {
        this.page = 1;
      } else {
        await this.submit(false)
      }
    }
  },
  methods: {
    async submit(new_search: bool) {
      if (new_search) {
        this.page = 1;
      }
      if (this.query) {
        this.loading = true;
        let collector_outputs = await this.summa.search(
            "page",
            {"query": {"match": {"value": this.query}}},
            [
              {"collector": {"top_docs": {
                "limit": 5,
                "offset": (this.page - 1) * 5,
                "snippets": { "text": 200, "title": 140 },
                "explain": true,
                "fields": [],
              }}},
              {"collector": {"count": {}}}
            ]
        );
        this.scored_documents = collector_outputs[0].collector_output.top_docs.scored_documents
        this.total_documents = collector_outputs[1].collector_output.count.count
        this.loading = false;
        this.stats_store.set_stats(await this.summa.stats());
      } else {
        this.scored_documents = []
      }
      this.$router.push({ name: 'search', query: { q: this.query, p: this.page } })
    },
  },
});
</script>

<template lang="pug">
div
  form
    div.input-group
      input.form-control.form-control-lg(v-model="query" type="text" placeholder="query")
      button.btn.btn-dark(type="submit" @click.stop.prevent="submit(true)") Search
  i.small.ms-3(v-if="total_documents > 0") {{ total_documents }} found
  .search-list.mt-4(v-if="!loading")
    search-list(:scored_documents='scored_documents')
    nav(v-if="scored_documents.length > 0")
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

<style scoped>
.search-list {
  padding-top: 15px;
  padding-bottom: 15px;
}
</style>
