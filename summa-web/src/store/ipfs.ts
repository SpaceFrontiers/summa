import { ref } from "vue";
import { defineStore } from "pinia";

export const useStore = defineStore("stats", () => {
  const stats = ref({});

  function set_stats(new_stats: any) {
    stats.value = new_stats;
  }
  return { stats };
});
