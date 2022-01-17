import init, { WebIndexRegistry } from "../pkg";
import {IndexAttributes, RemoteEngineConfig} from "./configs";

export class IndexQuery {
  index_alias: string
  query: Object;
  collectors: Object[];
  constructor(index_alias: string, query: Object, collectors: Object[]) {
    this.index_alias = index_alias;
    this.query = query;
    this.collectors = collectors
  }
}

export class WebIndexService {
  registry?: WebIndexRegistry;

  async setup(init_url: string, threads: number) {
    await init(init_url, new WebAssembly.Memory({ initial: 4096, maximum: 16384, shared: true }));
    this.registry = new WebIndexRegistry();
    await this.registry.setup(threads);
  }
  async add(remote_engine_config: RemoteEngineConfig, cb?: (tracker_event: object) => void): Promise<IndexAttributes> {
    console.log('js_add', remote_engine_config, cb);
    let add_operation = this.registry!.add(remote_engine_config);
    if (cb) {
      await add_operation.tracker().add_subscriber((tracker_event: object) => {
        cb(tracker_event)
      });
    }
    return await add_operation.execute();
  }
  async delete(index_name: string) {
    return await this.registry!.delete(index_name)
  }
  async search(index_queries: IndexQuery[], cb?: (tracker_event: object) => void) {
    let search_operation = this.registry!.search(index_queries);
    if (cb) {
      await search_operation.tracker().add_subscriber((tracker_event: object) => {
        cb(tracker_event)
      });
    }
    return await search_operation.execute();
  }
  async warmup(index_name: string, cb?: (tracker_event: object) => void) {
    let warmup_operation = this.registry!.warmup(index_name);
    if (cb) {
      await warmup_operation.tracker().add_subscriber((tracker_event: object) => {
        cb(tracker_event)
      });
    }
    return await warmup_operation.execute();
  }
  async index_document(index_name: string, document: string) {
    return await this.registry!.index_document(index_name, document)
  }
  async commit(index_name: string) {
    return await this.registry!.commit(index_name)
  }
}
