import { defineStore } from "pinia";
import type { WebIndexCoordinate } from "summa-wasm/web-index";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import * as localforage from "localforage";

Map.prototype.toJSON = function () {
  return [...this];
};

export const useWebIndexStore = defineStore("web_index_store", {
  state: () => {
    return {
      coordinates: new Map<IPFSPath, WebIndexCoordinate>(),
      names: new Map<String, [IPFSPath, IPFSPath]>(),
    };
  },
  actions: {
    async add(
      name: String,
      ipns_path: IPFSPath,
      ipfs_path: IPFSPath,
      web_index_coordinate: WebIndexCoordinate
    ) {
      this.coordinates.set(ipns_path, web_index_coordinate);
      this.names.set(name, [ipns_path, ipfs_path]);
      await useWebIndexStore().save();
    },
    async drop() {
      await localforage.clear();
    },
    is_empty() {
      return this.coordinates.size == 0;
    },
    async load() {
      this.coordinates = new Map(
        JSON.parse((await localforage.getItem("web_index:coordinates")) || "[]")
      );
      this.names = new Map(
        JSON.parse((await localforage.getItem("web_index:names")) || "[]")
      );
    },
    async save() {
      await localforage.setItem(
        "web_index:coordinates",
        JSON.stringify([...this.coordinates.entries()])
      );
      await localforage.setItem(
        "web_index:names",
        JSON.stringify([...this.names.entries()])
      );
    },
  },
});
