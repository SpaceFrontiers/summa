<template lang="pug">
div(style="transform: rotate(0);")
  h6.ms-1(v-html="title")
  a.stretched-link(:href="mono_link")
.fst-italic.small(v-if="authors") {{ authors }}
.mt-2.small(v-if="snippet", v-html="snippet")
.clearfix.small
  .text-muted.mt-2.float-end
    | {{ formatted_date }} | {{ format_bytes(filesize) }} | {{ extension }}
</template>

<script lang="ts">
// @ts-nocheck
import { defineComponent } from "vue";
import { ipfs_url } from "@/services/ipfs";
import { format_bytes } from "@/utils";

export default defineComponent({
  name: "NexusBook",
  props: {
    scored_document: {
      type: Object,
      required: true,
    },
  },
  methods: {
    format_bytes: format_bytes,
  },
  computed: {
    document() {
      return JSON.parse(this.scored_document["document"]);
    },
    authors() {
      let authors = "";
      if (this.document["authors"] !== undefined) {
        authors += this.document["authors"].join("; ");
      }
      return authors;
    },
    extension() {
      let extension = "pdf" as String;
      if (this.document["extension"] !== undefined) {
        extension = this.document["extension"];
      }
      return extension;
    },
    filename() {
      return (
        this.document["title"]
          .toLowerCase()
          .replaceAll(/\W/gu, " ")
          .replaceAll(/\s+/gu, " ")
          .replaceAll(" ", "-") +
        "." +
        this.extension
      );
    },
    filesize() {
      return this.document["filesize"];
    },
    formatted_date() {
      return this.document["year"];
    },
    mono_link() {
      if (this.document["ipfs_multihashes"] !== undefined) {
        return `${ipfs_url}/ipfs/${this.document["ipfs_multihashes"][0]}?filename=${this.filename}&download=true`;
      }
      return this.telegram_link;
    },
    snippet() {
      if (this.document.description === undefined) {
        return "";
      }
      let description = this.scored_document.snippets.description.html;
      if (description.length === 0) {
        description = this.document.description.substring(0, 400);
        if (this.document.description.length > 400) {
          description += "...";
        }
      } else {
        let encoder = new TextEncoder();
        const original_length = encoder.encode(
          this.document.description
        ).length;
        const snippet_length =
          this.scored_document.snippets.description.fragment.length;
        if (original_length > snippet_length) {
          description += "...";
        }
        if (description[0] == description[0].toLowerCase()) {
          description = "..." + description;
        }
      }
      return description;
    },
    telegram_link() {
      let telegram_query = btoa(`id:${this.document["id"]}`);
      return `https://t.me/libgen_scihub_3_bot?start=${telegram_query}`;
    },
    title() {
      let title = this.scored_document.snippets.title.html;
      if (title.length === 0) {
        title = this.document.title || "No title";
      } else {
        let encoder = new TextEncoder();
        const original_length = encoder.encode(this.document.title).length;
        const snippet_length =
          this.scored_document.snippets.title.fragment.length;
        if (original_length > snippet_length) {
          title += "...";
        }
      }
      title = "📚 " + title;
      return title;
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
