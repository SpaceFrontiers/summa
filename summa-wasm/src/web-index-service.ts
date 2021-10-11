import * as Comlink from "comlink";
import init, { cache_metrics, init_thread_pool, WebIndexInner } from "./";
import { WebIndexCoordinate } from "./web-index";

const web_index_service = {
  registry: new Map<String, WebIndexInner>(),
  async setup(status_callback: any, threads: number) {
    this.status_callback = status_callback;
    this.status_callback("status", "setting workers...");
    await init();
    this.status_callback("status", "setting thread pool...");
    await init_thread_pool(threads);
  },
  async add_index(coordinate: WebIndexCoordinate) {
    const web_index = new WebIndexInner(
      coordinate.method,
      coordinate.url_template,
      coordinate.headers_template,
      coordinate.files,
      this.status_callback,
    );
    if (this.registry.has(web_index.name)) {
      this.registry.get(web_index.name)!.free();
    }
    this.registry.set(web_index.name, web_index);
    return this.metadata(web_index.name);
  },
  async search(name: String, query: Object, collectors: Object[]) {
    return await this.registry.get(name)!.search(name, query, collectors);
  },
  async free(name: String) {
    this.registry(name)!.free();
    this.registry.delete(name);
  },
  async warmup(name: String) {
    this.status_callback(`warming up ${name}...`);
    await this.registry.get(name)!.warmup();
  },
  async metadata(name: String) {
    const web_index = this.registry.get(name)!;
    return {
      name: web_index.name,
      description: web_index.description,
      unixtime: web_index.unixtime
    }
  },
  async cache_metrics() {
    return await cache_metrics()
  }
}
export type WebIndexService = typeof web_index_service;
Comlink.expose(web_index_service);
