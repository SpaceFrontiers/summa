<template lang="pug">
.d-flex.justify-content-center
  .spinner-border(role="status")
.d-flex.justify-content-center.m-5
  .small(v-if="label") {{ label }}
</template>

<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "IsLoading",
  props: {
    label: {
      type: String,
    },
  },
});
</script>
