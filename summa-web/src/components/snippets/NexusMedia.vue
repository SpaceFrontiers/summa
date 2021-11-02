<template lang="pug">
div(style="transform: rotate(0);")
  h6.ms-1(v-html="title")
  a.stretched-link(:href="mono_link")
.fst-italic.small(v-if="category") {{ category }}
.mt-2.small(v-if="snippet", v-html="snippet")
.clearfix.small
  .text-muted.mt-2.float-end
    | {{ formatted_date }} | {{ format_bytes(filesize) }}
</template>

<script lang="ts">
// @ts-nocheck
import { defineComponent } from "vue";
import { format_bytes } from "@/utils";

export default defineComponent({
  name: "NexusMedia",
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
    category() {
      return this.document.category
    },
    document() {
      return JSON.parse(this.scored_document.document);
    },
    filesize() {
      return this.document["size"];
    },
    mono_link() {
      return `magnet:?xt=urn:btih:${
        this.document.torrent_hash
      }&tr=${encodeURIComponent(this.tracker)}`;
    },
    snippet() {
      if (this.document.content === undefined) {
        return "";
      }
      let content = this.scored_document.snippets.content.html;
      if (content.length === 0) {
        content = this.document.content.substring(0, 400);
        if (this.document.content.length > 400) {
          content += "...";
        }
      } else {
        let encoder = new TextEncoder();
        const original_length = encoder.encode(
          this.document.content
        ).length;
        const snippet_length =
          this.scored_document.snippets.content.fragment.length;
        if (original_length > snippet_length) {
          content += "...";
        }
        if (content[0] == content[0].toLowerCase()) {
          content = "..." + content;
        }
      }
      return content;
    },
    formatted_date() {
      return new Date(this.document.registered_at * 1000).toLocaleDateString();
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
      title = "🧲 " + title;
      return title;
    },
    tracker() {
      switch (this.document.torrent_tracker_id) {
        case 1:
          return "http://bt.t-ru.org/ann?magnet";
        default:
          return `http://bt${this.document.torrent_tracker_id}.t-ru.org/ann?magnet`;
      }
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
