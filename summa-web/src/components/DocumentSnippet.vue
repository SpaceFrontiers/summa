<template lang="pug">
.container
  h6.ms-1(v-html="title")
  p.small(v-if="snippet", v-html="snippet")
hr
</template>

<script lang="ts">
// @ts-nocheck
import {defineComponent} from "vue";

export default defineComponent({
  name: "DocumentSnippet",
  props: {
    scored_document: {
      type: Object,
      required: true,
    },
  },
  computed: {
    document() {
      return JSON.parse(this.scored_document['document'])
    },
    text() {
      return this.document.text
    },
    snippet() {
      let snippet = this.scored_document.snippets.text.html
      if (this.document.text !== undefined) {
        let encoder = new TextEncoder();
        if (encoder.encode(this.document.text).length > this.scored_document.snippets.text.fragment.length) {
          snippet += '...'
        }
        if (snippet[0] == snippet[0].toLowerCase()) {
          snippet = '...' + snippet
        }
      }
      return snippet
    },
    title() {
      let title = this.scored_document.snippets.title.html
      let encoder = new TextEncoder();
      if (encoder.encode(this.document.title).length > this.scored_document.snippets.title.fragment.length) {
        title += '...'
      }
      return title
    },
  }
});
</script>

<style scoped lang="scss">
li {
  padding-bottom: 15px;
  padding-left: 0;
  &:after {
    content: none;
  }
}
</style>
