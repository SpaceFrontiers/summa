import * as Comlink from "comlink";
import type { WebIndexService as WebIndexServiceWasm } from "summa-wasm/web-index-service";
import { useWebIndexStore } from "@/store/web_index";
import { ref, toRaw } from "vue";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import {
  create_web_index_ipfs_coordinate,
  WebIndexCoordinate,
} from "summa-wasm/web-index";
import { ipfs, ipfs_url } from "./ipfs";
import type { Store } from "pinia";
import type { Remote } from "comlink";
import localforage from "localforage";

const default_indices = [
  "/ipns/k51qzi5uqu5dl73ko65n1dhgj1uxkcomabckudr8at1n7g6jm87upcbvvn6xdt",
];

export type StatusCallback = (type: string, message: string) => void;
export class WebIndexService {
  status_callback: StatusCallback;
  web_index_store: Store<
    "web_index_store",
    {
      names: Map<String, [IPFSPath, IPFSPath]>;
      coordinates: Map<IPFSPath, WebIndexCoordinate>;
    },
    {},
    {
      add(
        name: String,
        ipns_path: IPFSPath,
        ipfs_path: IPFSPath,
        web_index_coordinate: WebIndexCoordinate
      ): Promise<void>;
      drop(): Promise<void>;
      load(): Promise<void>;
      save(): Promise<void>;
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
        Comlink.proxy(this.status_callback),
        options.num_threads
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
    for (const web_index_coordinate of this.web_index_store.coordinates.values()) {
      await this.web_index_service_worker.add_index(
        toRaw(web_index_coordinate)
      );
    }
  }
  async add_index(
    ipns_path: IPFSPath,
    ipfs_path: IPFSPath,
    web_index_coordinate: WebIndexCoordinate
  ) {
    const metadata = await this.web_index_service_worker.add_index(
      web_index_coordinate
    );
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    await this.web_index_store.add(
      metadata.name,
      ipns_path,
      ipfs_path,
      web_index_coordinate
    );
  }
  async free(name: string) {
    await this.web_index_service_worker.free(name);
  }
  async search(name: String, query: Object, collectors: Object[]) {
    const response = await this.web_index_service_worker.search(
      name,
      query,
      collectors
    );
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    return response;
  }
  async resolve(ipfs_path: IPFSPath): Promise<WebIndexCoordinate> {
    const files = await ipfs.ls(ipfs_path);
    return create_web_index_ipfs_coordinate(
      ipfs_url,
      ipfs_path as string,
      files
    );
  }
  is_empty() {
    return this.web_index_store.is_empty();
  }
  async install_defaults() {
    for (const ipns_path of default_indices) {
      this.status_callback("status", `resolving ${ipns_path}...`);
      const ipfs_path = await ipfs.resolve(ipns_path);
      this.status_callback("status", `resolving files...`);
      const web_index_coordinate = await this.resolve(ipfs_path);
      await this.add_index(ipns_path, ipfs_path, web_index_coordinate);
    }
  }
  async metadata(name: string) {
    const web_index_metadata = await this.web_index_service_worker.metadata(
      name
    );
    const [ipns_path, ipfs_path] = this.web_index_store.names.get(name)!;
    const web_index_coordinate =
      this.web_index_store.coordinates.get(ipns_path)!;
    return {
      ipns_path: ipns_path,
      ipfs_path: ipfs_path,
      enabled: web_index_coordinate.enabled,
      name: web_index_metadata.name,
      description: web_index_metadata.description,
      unixtime: web_index_metadata.unixtime,
    };
  }
  async metadatas() {
    const result = [];
    for (const name of this.web_index_store.names.keys()) {
      result.push(await this.metadata(name as string));
    }
    return result;
  }
}

export const cache_metrics = ref({
  requests: 0,
  bytes_received: 0,
});
