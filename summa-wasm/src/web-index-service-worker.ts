import * as Comlink from "comlink";
import init, { cache_metrics, init_thread_pool, WebIndexRegistry } from "../pkg";
import { NetworkConfig } from "./configs";

export type StatusCallback = (type: string, message: string) => void;
export class IndexQuery {
  index_name: string
  query: Object;
  collectors: Object[];
  constructor(index_name: string, query: Object, collectors: Object[]) {
    this.index_name = index_name;
    this.query = query;
    this.collectors = collectors
  }
}

export class WebIndexServiceWorker {
  registry?: WebIndexRegistry;
  async setup(init_url: string, threads: number, status_callback?: StatusCallback) {
    if (!status_callback) {
      status_callback = (type: string, message: string) => console.log(type, message)
    }
    status_callback("status", "setting workers...");
    await init(init_url);
    this.registry = new WebIndexRegistry(threads > 0);
    if (threads > 0) {
      status_callback("status", "setting thread pool of size " + threads.toString() + "...");
      await init_thread_pool(threads);
    }
  }
  async add(network_config: NetworkConfig): Promise<Object> {
    return await this.registry!.add(network_config)
  }
  async delete(index_name: string) {
    return await this.registry!.delete(index_name)
  }
  async search(index_queries: IndexQuery[]) {
    return await this.registry!.search(index_queries)
  }
  async cache_metrics() {
    return await cache_metrics()
  }
  async warmup(index_name: string) {
    return await this.registry!.warmup(index_name);
  }
}
export const web_index_service_worker = new WebIndexServiceWorker();
Comlink.expose(web_index_service_worker);
