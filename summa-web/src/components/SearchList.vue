<template lang="pug">
div(v-for="scored_document in scored_documents" v-bind:key="scored_document['position']")
  document-snippet(:scored_document="scored_document")
</template>

<script lang="ts">
import { defineComponent, type PropType } from "vue";
import DocumentSnippet from "@/components/DocumentSnippet.vue";

export default defineComponent({
  name: "SearchList",
  components: { DocumentSnippet },
  props: {
    scored_documents: {
      type: Array as PropType<
        {
          document: {
            id: number[];
            title: string[];
            ipfs_multihashes: string[];
          };
        }[]
      >,
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
