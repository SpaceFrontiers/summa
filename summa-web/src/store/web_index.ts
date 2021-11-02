import { defineStore } from "pinia";
import type { NetworkConfig } from "summa-wasm/network-config";
import * as localforage from "localforage";

Map.prototype.toJSON = function () {
  return [...this];
};

export class IndexConfig {
  enabled: boolean;
  network_config: NetworkConfig;
  constructor(network_config: NetworkConfig) {
    this.enabled = true;
    this.network_config = network_config;
  }
}

export const useWebIndexStore = defineStore("web_index_store", {
  state: () => {
    return {
      index_configs: new Map<String, IndexConfig>(),
    };
  },
  actions: {
    async add(name: String, network_config: NetworkConfig) {
      this.index_configs.set(name, new IndexConfig(network_config));
      await useWebIndexStore().save();
    },
    async delete(name: String) {
      this.index_configs.delete(name);
      await useWebIndexStore().save();
    },
    async switch(name: String) {
      const index_config = this.index_configs.get(name)!;
      index_config.enabled = !index_config.enabled;
      await useWebIndexStore().save();
    },
    async drop() {
      await localforage.clear();
    },
    is_empty() {
      return this.index_configs.size == 0;
    },
    async load() {
      this.index_configs = new Map(
        JSON.parse(
          (await localforage.getItem("web_index:index_configs")) || "[]"
        )
      );
      for (const index_config of this.index_configs.values()) {
        index_config.network_config.files = new Map(
          index_config.network_config.files
        );
      }
    },
    async save() {
      await localforage.setItem(
        "web_index:index_configs",
        JSON.stringify([...this.index_configs.entries()])
      );
    },
  },
});
