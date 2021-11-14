<template lang="pug">
.font-monospace.small
  h5 How to help us?
  p.mt-4 Cryptos:
  ul
    li
      b BTC:&nbsp;
      text 3HooXUqJnZ4Ad8AGeqfSZ5QZQE72ZaZgY6
    li
      b ETH:&nbsp;
      text 0x009AeabF4aeDe417d324077E7858956e6d0962D6
    li
      b XMR:&nbsp;
      text 88NyLYKiuWPaNUoJrRTZQxNTT9JevoGHjacLaJp7F4UBVt9K6x3i7Khhgmap2bXRz6UczvNpu6XLS9GxoQ2RWdHQ3Kv4oMA
  h5 Development
  p.mt-4
    text GitHub:&nbsp;
    a(href="https://github.com/izihawa/summa") izihawa/summa
  p
    text TG Group:&nbsp;
    a(href="https://t.me/summa_search") @summa_search
  h5 Nexus Databases Development
  p.mt-4
    text TG Channel:&nbsp;
    a(href="https://t.me/nexus_search") @nexus_search
  p
    text TG Group:&nbsp;
    a(href="https://t.me/joinchat/SKgGpCnrzMdkMRxJ") SKgGpCnrzMdkMRxJ
</template>
<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "AboutView",
});
</script>
