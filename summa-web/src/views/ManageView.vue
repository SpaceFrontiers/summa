<template lang="pug">
h4
  | My god... it's full of stars!
img.img-fluid(:src="stars_url")
</template>
<script lang="ts">
import {defineComponent} from "vue";
import starsJpegUrl from '@/assets/stars.jpeg'

export default defineComponent({
  name: "AboutView",
  data() {
    return {
      stars_url: starsJpegUrl
    }
  }
})

</script>

<style>
@media (min-width: 1024px) {
  .about {
    min-height: 100vh;
    display: flex;
    align-items: center;
  }
}
</style>
