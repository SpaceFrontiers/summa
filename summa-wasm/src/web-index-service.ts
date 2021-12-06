import init, { cache_metrics, init_thread_pool, WebIndexRegistry } from "../pkg";
import { RemoteEngineConfig } from "./configs";

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

export class WebIndexService {
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
      status_callback("status", "");
    }
  }
  async add(index_engine: { config: { remote: RemoteEngineConfig } | { memory: Object }}): Promise<string> {
    return await this.registry!.add(index_engine)
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
  async index_document(index_name: string, document: string) {
    return await this.registry!.index_document(index_name, document)
  }
  async commit(index_name: string) {
    return await this.registry!.commit(index_name)
  }
}
