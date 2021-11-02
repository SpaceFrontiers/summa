import * as Comlink from "comlink";
import init, { cache_metrics, init_thread_pool, WebIndexRegistry } from "./";

export type IndexPayload = {
  name: String;
  description: String;
  unixtime: Number;
};

const web_index_service = {
  async setup(threads: number, status_callback: any) {
    this.status_callback = status_callback;
    this.status_callback("status", "setting workers...");
    await init();
    this.registry = new WebIndexRegistry();
    this.status_callback("status", "setting thread pool of size " + threads.toString() + "...");
    await init_thread_pool(threads);
  },
  async add(network_config: any) {
    network_config.caching_config = {
      cache_size: 32 * 1024 * 1024,
      chunk_size: 2 ** 16
    }
    return await this.registry.add(network_config)
  },
  async delete(index_name: String) {
    return await this.registry.delete(index_name)
  },
  async search(index_names: String[], query: any, collectors: any) {
    return await this.registry.search(index_names, query, collectors)
  },
  async get_index_payload(index_name: String): IndexPayload {
    return await this.registry.get_index_payload(index_name)
  },
  async cache_metrics() {
    return await cache_metrics()
  }
}
export type WebIndexService = typeof web_index_service;
Comlink.expose(web_index_service);
