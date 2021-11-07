import * as Comlink from "comlink";
import type {
  IndexPayload,
  IndexQuery,
  WebIndexService as WebIndexServiceWasm,
} from "summa-wasm/web-index-service";
import { db, IndexConfig } from "@/database";
import { ref, toRaw } from "vue";
import type { Remote } from "comlink";
import {
  ipfs,
  ipfs_hostname,
  ipfs_http_protocol,
} from "@/services/ipfs";
import { NetworkConfig } from "summa-wasm/network-config";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";

const default_index_seeds = [
  {
    ipns_path: "/ipns/nexus-books.eth/",
    is_enabled: true,
  },
  {
    ipns_path: "/ipns/nexus-media.eth/",
    is_enabled: false,
  },
];

export type StatusCallback = (type: string, message: string) => void;
export class WebIndexService {
  status_callback: StatusCallback;
  web_index_service_worker: Remote<WebIndexServiceWasm>;

  constructor() {
    this.status_callback = (type: string, message: string) =>
      console.log(type, message);
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
      await this.web_index_service_worker.setup(
        options.num_threads,
        Comlink.proxy(this.status_callback)
      );
      await this.load_from_store();
    } catch (e) {
      console.error("Dropping stored data due to error: ", e);
      await db.index_configs.clear();
      throw e;
    }
    try {
      if (await this.is_empty()) {
        await this.install_defaults();
      }
    } catch (e) {
      console.error(e);
      return false;
    }
    return true;
  }
  async load_from_store() {
    for (const index_config of await db.index_configs.toArray()) {
      await this.web_index_service_worker.add(
        toRaw(index_config.network_config)
      );
    }
  }
  async add_index(index_seed: {
    ipns_path: IPFSPath;
    is_enabled: boolean;
  }): Promise<IndexPayload> {
    this.status_callback("status", `resolving ${index_seed.ipns_path}...`);
    const ipfs_path = await ipfs.resolve(
      (index_seed.ipns_path as string).split("/")[2]
    );
    const ipfs_hash = ipfs_path.split("/")[2] as string;
    this.status_callback("status", `resolving files...`);
    const files = await ipfs.ls(ipfs_hash);
    const network_config = new NetworkConfig(
      "GET",
      `${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/{file_name}`,
      [{ name: "range", value: "bytes={start}-{end}" }],
      files
    );
    const index_payload = await this.web_index_service_worker.add(
      network_config
    );
    const index_config = new IndexConfig(
      index_seed.is_enabled,
      false,
      index_payload,
      index_seed.ipns_path,
      network_config
    );
    await index_config.save();
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    return index_payload;
  }
  async delete_index(index_name: string) {
    await this.web_index_service_worker.delete(index_name);
    await db.index_configs.delete(index_name);
  }
  async search(index_queries: IndexQuery[]) {
    const response = await this.web_index_service_worker.search(index_queries);
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    return response;
  }
  async is_empty() {
    return (await db.index_configs.count()) == 0;
  }
  async install_defaults() {
    return await Promise.all(
      default_index_seeds.map((default_index_seed) =>
        this.add_index(default_index_seed)
      )
    );
  }
}

export const cache_metrics = ref({
  requests: 0,
  bytes_received: 0,
});
