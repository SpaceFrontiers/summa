<template lang="pug">
.font-monospace
  h5.mb-3 Access Point for Knowledge Network
  div
    b.me-2 phase:
    text 2b
  div
    b.me-2 epoch:
    text {{ trantorianTime() }}gs ({{ now() }} Planetary Local Time)
  h6.mb-3.mt-3 Log (automatically translated to the most spread Earth language)
  ul
    li
      b 2022-10
      text  - started phase2b, initiated knowledge transferring to immutable carrier
    li
      b 2022-08
      text  - locals are adopting novel approaches. seems that limits and recently mentioned obstacles are started to disappear
    li
      b 2022-02
      text  - ignorance still present. must double efforts.
    li
      b 2020-03
      text  - started phase2, continue to increase availability of knowledge using local technological tools
    li
      b 2017-05
      text  - widespread adoption of knowledge tools. observed general increase of stored knowledge in neural networks of local inhabitants
    li
      b 2011-09
      text  - locals have created exchanging hub for knowledge. raw, unprocessed. however we are moving to the right direction
    li
      b 2008-07
      text  - started phase1, explored target planet. found necessity to eliminate temporary obstacles for creation of cognitron. local inhabitants use primitive approaches to knowledge organizing

</template>
<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "AboutView",
  methods: {
    now() {
      const currentTime = new Date();
      return `${currentTime.getFullYear()}-${String(
        currentTime.getMonth() + 1
      ).padStart(2, "0")}`;
    },
    trantorianTime() {
      return 435196800;
    },
  },
});
</script>
