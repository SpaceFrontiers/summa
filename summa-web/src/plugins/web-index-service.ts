import * as Comlink from "comlink";
import type { IndexPayload, WebIndexService as WebIndexServiceWasm } from "summa-wasm/web-index-service";
import {IndexConfig, useWebIndexStore} from "@/store/web_index";
import { ref, toRaw } from "vue";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import { create_network_config, NetworkConfig } from "summa-wasm/network-config";
import { ipfs, ipfs_url } from "./ipfs";
import type { Store } from "pinia";
import type { Remote } from "comlink";

const default_indices = [
  "/ipns/k51qzi5uqu5dl73ko65n1dhgj1uxkcomabckudr8at1n7g6jm87upcbvvn6xdt",
  "/ipns/k51qzi5uqu5dkr0d4zdv93jrwvbbwqyr2snps0yff6htztbr1db16v19oaxiej",
];

export type StatusCallback = (type: string, message: string) => void;
export class WebIndexService {
  status_callback: StatusCallback;
  web_index_store: Store<
    "web_index_store",
    { index_configs: Map<String, IndexConfig> },
    {},
    {
      add(name: String, network_config: NetworkConfig): Promise<void>;
      drop(): Promise<void>;
      load(): Promise<void>;
      save(): Promise<void>;
      delete(name: String): Promise<void>;
      is_empty(): boolean;
    }
  >;
  web_index_service_worker: Remote<WebIndexServiceWasm>;

  constructor() {
    this.status_callback = (type: string, message: string) =>
      console.log(type, message);
    this.web_index_store = useWebIndexStore();
    this.web_index_service_worker = Comlink.wrap<WebIndexServiceWasm>(
      new Worker(
        new URL(
          "../../node_modules/summa-wasm/web-index-service.ts",
          import.meta.url
        ),
        { type: "module" }
      )
    );
  }
  async setup(options: { num_threads: number }) {
    try {
      await this.web_index_store.load();
      await this.web_index_service_worker.setup(
        options.num_threads,
        Comlink.proxy(this.status_callback)
      );
      await this.load_from_store();
    } catch (e) {
      console.error("Dropping stored data due to error: ", e);
      await this.web_index_store.drop();
      throw e;
    }
    try {
      if (this.is_empty()) {
        await this.install_defaults();
      }
    } catch (e) {
      console.error(e);
      return false;
    }
    return true;
  }
  async load_from_store() {
    for (const index_config of this.web_index_store.index_configs.values()) {
      await this.web_index_service_worker.add(toRaw(index_config.network_config));
    }
  }
  async add_index(network_config: NetworkConfig): IndexPayload {
    const index_payload = await this.web_index_service_worker.add(
      network_config
    );
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    await this.web_index_store.add(index_payload.name, network_config);
  }
  async delete_index(index_name: string) {
    await this.web_index_service_worker.delete(index_name);
    await this.web_index_store.delete(index_name);
  }
  async search(index_names: String[], query: Object, collectors: Object[]) {
    const response = await this.web_index_service_worker.search(
      index_names,
      query,
      collectors
    );
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    return response;
  }
  async resolve(ipfs_path: IPFSPath): Promise<NetworkConfig> {
    const files = await ipfs.ls(ipfs_path);
    return create_network_config(ipfs_url, ipfs_path as string, files);
  }
  is_empty() {
    return this.web_index_store.is_empty();
  }
  async install_defaults() {
    for (const ipns_path of default_indices) {
      await this.install_index(ipns_path);
    }
  }
  async install_index(ipns_path: IPFSPath) {
    this.status_callback("status", `resolving ${ipns_path}...`);
    const ipfs_path = await ipfs.resolve(ipns_path);
    this.status_callback("status", `resolving files...`);
    const network_config = await this.resolve(ipfs_path);
    await this.add_index(network_config);
  }
  async get_index_payload(index_name: String): IndexPayload {
    return await this.web_index_service_worker.get_index_payload(index_name)!;
  }
  async get_index_payloads() {
    const result = new Map();
    for (const index_name of this.web_index_store.index_configs.keys()) {
      result.set(
        index_name,
        await this.get_index_payload(index_name as string)
      );
    }
    return result;
  }
}

export const cache_metrics = ref({
  requests: 0,
  bytes_received: 0,
});
