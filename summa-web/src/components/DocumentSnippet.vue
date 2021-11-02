<template lang="pug">
.container
  nexus-book(v-if='scored_document.index_alias === "nexus_books"' :scored_document = "scored_document")
  nexus-media(v-if='scored_document.index_alias === "nexus_media"' :scored_document = "scored_document")
hr
</template>

<script lang="ts">
// @ts-nocheck
import { defineComponent } from "vue";
import NexusBook from "./snippets/NexusBook.vue";
import NexusMedia from "./snippets/NexusMedia.vue";


export default defineComponent({
  name: "DocumentSnippet",
  components: { NexusBook, NexusMedia },
  props: {
    scored_document: {
      type: Object,
      required: true,
    },
  },
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
