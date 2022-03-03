export type IndexEngineConfigType = {remote: RemoteEngineConfig} | {memory: MemoryEngineConfig};

export class IndexEngineConfig {
  config: IndexEngineConfigType
  constructor(config: IndexEngineConfigType) {
    this.config = config;
  }
}

export class MemoryEngineConfig {
  schema: string;
  constructor(schema: string) {
    this.schema = schema
  }
}

export class RemoteEngineConfig {
  method: string;
  url_template: string;
  headers_template: Map<string, string> | null;
  cache_config?: CacheConfig;
  timeout_ms?: number;
  constructor(
    method: string,
    url_template: string,
    headers_template: Map<string, string> | null,
    cache_config?: CacheConfig,
    timeout_ms?: number,
  ) {
    this.method = method
    this.url_template = url_template
    this.headers_template = headers_template;
    this.cache_config = cache_config;
    this.timeout_ms = timeout_ms;
  }
}

export class CacheConfig {
  cache_size: number;
  constructor(
    cache_size: number
  ) {
    this.cache_size = cache_size
  }
}

export class IndexAttributes {
  created_at: number;
  unique_fields?: string[];
  default_fields: string[];
  multi_fields: string[];
  default_index_name?: string;
  description?: string;
  default_snippets?: string;
  constructor(
    created_at: number,
    default_fields: string[],
    multi_fields: string[],
    unique_fields?: string[],
    default_index_name?: string,
    description?: string,
    default_snippets?: string,
  ) {
    this.created_at = created_at
    this.default_fields = default_fields
    this.multi_fields = multi_fields
    this.unique_fields = unique_fields
    this.default_index_name = default_index_name
    this.description = description
    this.default_snippets = default_snippets
  }
}
