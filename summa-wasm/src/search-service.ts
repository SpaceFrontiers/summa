import { IndexAttributes, RemoteEngineConfig } from "./configs";

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

export interface ISearchService {
  add(remote_engine_config: RemoteEngineConfig, index_name?: string): Promise<IndexAttributes>;
  delete(index_name: string): Promise<void>;
  search(index_queries: IndexQuery[]): Promise<object[]>;
  warmup(index_name: string): Promise<void>;
  index_document(index_name: string, document: string): Promise<void>;
  commit(index_name: string): Promise<void>;
}
