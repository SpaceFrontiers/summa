import init, {setup_logging, WrappedIndexRegistry} from "../pkg";
import {IndexAttributes, IndexEngineConfig} from "./grpc-web/index_service";
import {SearchRequest} from "./grpc-web/query";

export interface IIndexRegistry {
  add(index_name: string, index_engine_config: IndexEngineConfig): Promise<IndexAttributes>;
  delete(index_name: string): Promise<void>;
  search(search_request: SearchRequest): Promise<object[]>;
  search_by_binary_proto(search_request_proto_bytes: Uint8Array): Promise<object[]>;
  warmup(index_name: string): Promise<void>;
  index_document(index_name: string, document: string): Promise<void>;
  commit(index_name: string): Promise<void>;
  get_index_field_names(index_name: string): Promise<string[]>;
}

export type IndexRegistryOptions = {
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
  logging_level: "info",
  memory_config: is_mobile() ? { initial: 1024, maximum: 8192, shared: true } : { initial: 2048, maximum: 65536, shared: true }
}

export class IndexRegistry implements IIndexRegistry {
  registry?: WrappedIndexRegistry;

  async setup(
      init_url: string,
      options: IndexRegistryOptions = default_options,
  ) {
    let actual_options = Object.assign({}, default_options, options);
    console.log('Memory config:', actual_options.memory_config);
    try {
      await init(init_url, new WebAssembly.Memory(actual_options.memory_config!));
    } catch (e) {
      await init(init_url + "?force", new WebAssembly.Memory(actual_options.memory_config!));
    }
    await setup_logging(actual_options.logging_level!);
    this.registry = new WrappedIndexRegistry();
  }

  async add(index_name: string, index_engine_config: IndexEngineConfig): Promise<IndexAttributes> {
    return await this.registry!.add(index_name, index_engine_config);
  }
  async delete(index_name: string) {
    return await this.registry!.delete(index_name)
  }
  async search(search_request: SearchRequest): Promise<object[]> {
    return await this.registry!.search(search_request);
  }
  async search_by_binary_proto(search_request_bytes_proto: Uint8Array): Promise<object[]> {
    return await this.registry!.search_by_binary_proto(search_request_bytes_proto);
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
  async get_index_field_names(index_name: string): Promise<string[]> {
    return await this.registry!.get_index_field_names(index_name);
  }
}
