import * as Comlink from "comlink";
import init, { cache_metrics, init_thread_pool, WebIndexRegistry } from "./";

export type IndexPayload = {
  name: string;
  description: string;
  unixtime: number;
};

export type IndexQuery = {
  collectors: Object;
  query: Object;
  index_name: string
}

const web_index_service = {
  async setup(threads: number, status_callback: any) {
    this.status_callback = status_callback;
    this.status_callback("status", "setting workers...");
    await init();
    this.registry = new WebIndexRegistry(threads > 0);
    if (threads > 0) {
      this.status_callback("status", "setting thread pool of size " + threads.toString() + "...");
      await init_thread_pool(threads);
    }
  },
  async add(network_config: any) {
    network_config.chunked_cache_config = {
      cache_size: 128 * 1024 * 1024,
      chunk_size: 16 * 1024,
    }
    return await this.registry.add(network_config)
  },
  async delete(index_name: string) {
    return await this.registry.delete(index_name)
  },
  async search(index_queries: IndexQuery[]) {
    return await this.registry.search(index_queries)
  },
  async get_index_payload(index_name: string): Promise<IndexPayload> {
    return await this.registry.get_index_payload(index_name)
  },
  async cache_metrics() {
    return await cache_metrics()
  },
  async warmup(index_name: string) {
    return await this.registry.warmup(index_name);
  }
}
export type WebIndexService = typeof web_index_service;
Comlink.expose(web_index_service);
