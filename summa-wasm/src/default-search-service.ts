import init, {reserve_heap, setup_logging, WebIndexRegistry} from "../pkg";
import {IndexAttributes, RemoteEngineConfig} from "./configs";
import { IndexQuery, ISearchService } from "./search-service";

export class DefaultSearchService implements ISearchService {
  registry?: WebIndexRegistry;

  async setup(init_url: string, threads: number) {
    await init(init_url, new WebAssembly.Memory({ initial: 4096, maximum: 16384, shared: true }));
    await setup_logging("info");
    await reserve_heap();

    this.registry = new WebIndexRegistry();
    await this.registry.setup(threads);
  }

  async add(remote_engine_config: RemoteEngineConfig, index_name?: string): Promise<IndexAttributes> {
    return await this.registry!.add(remote_engine_config, index_name);
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
