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
  extract_terms(index_name: string, field_name: string, limit: number, start_from?: string): Promise<string[]>;
  get_index_field_names(index_name: string): Promise<string[]>;
}

export type IndexRegistryOptions = {
  num_threads?: number,
  logging_level?: string
  memory_config?: WebAssembly.MemoryDescriptor
}

function is_mobile() {
  return (
      navigator.userAgent.match(/Android/i)
      || navigator.userAgent.match(/webOS/i)
      || navigator.userAgent.match(/iPhone/i)
      || navigator.userAgent.match(/iPad/i)
      || navigator.userAgent.match(/iPod/i)
      || navigator.userAgent.match(/BlackBerry/i)
      || navigator.userAgent.match(/Windows Phone/i)
  )
}

export const default_options: IndexRegistryOptions = {
  num_threads: Math.ceil(navigator.hardwareConcurrency / 2),
  logging_level: "info",
  memory_config: is_mobile() ? { initial: 1024, maximum: 8192, shared: true } : { initial: 8192, maximum: 65536, shared: true }
}

export class IndexRegistry implements IIndexRegistry {
  registry?: WrappedIndexRegistry;

  async setup(
      init_url: string,
      options: IndexRegistryOptions = default_options,
  ) {
    let actual_options = Object.assign({}, default_options, options);
    console.log('Memory config:', actual_options.memory_config);
    await init(init_url, new WebAssembly.Memory(actual_options.memory_config!));
    await setup_logging(actual_options.logging_level!);

    this.registry = new WrappedIndexRegistry();
    await this.registry.setup(actual_options.num_threads!);
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
  async extract_terms(index_name: string, field_name: string, limit: number, start_from?: string): Promise<string[]> {
    return await this.registry!.extract_terms(index_name, field_name, limit, start_from);
  }
  async get_index_field_names(index_name: string): Promise<string[]> {
    return await this.registry!.get_index_field_names(index_name);
  }
}
