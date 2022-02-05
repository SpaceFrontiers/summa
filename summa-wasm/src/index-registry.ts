import {IndexAttributes, IndexEngineConfig} from "./configs";
import init, {reserve_heap, setup_logging, WrappedIndexRegistry} from "../pkg";

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

export interface IIndexRegistry {
  add(index_engine_config: IndexEngineConfig, index_name?: string): Promise<IndexAttributes>;
  delete(index_name: string): Promise<void>;
  search(index_queries: IndexQuery[]): Promise<object[]>;
  warmup(index_name: string): Promise<void>;
  index_document(index_name: string, document: string): Promise<void>;
  commit(index_name: string): Promise<void>;
}

export class IndexRegistry implements IIndexRegistry {
  registry?: WrappedIndexRegistry;

  async setup(init_url: string, threads: number) {
    await init(init_url, new WebAssembly.Memory({ initial: 4096, maximum: 16384, shared: true }));
    await setup_logging("info");
    await reserve_heap();

    this.registry = new WrappedIndexRegistry();
    await this.registry.setup(threads);
  }

  async add(index_engine_config: IndexEngineConfig, index_name?: string): Promise<IndexAttributes> {
    return await this.registry!.add(index_engine_config, index_name);
  }
  async delete(index_name: string) {
    return await this.registry!.delete(index_name)
  }
  async search(index_queries: IndexQuery[]): Promise<object[]> {
    return await this.registry!.search(index_queries);
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
