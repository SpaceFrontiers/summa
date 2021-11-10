export class NetworkConfig {
  method: string;
  url_template: string;
  headers_template: {name: string, value: string}[] | null;
  chunked_cache_config?: ChunkedCacheConfig;
  constructor(
    method: string,
    url_template: string,
    headers_template: {name: string, value: string}[] | null,
    chunked_cache_config?: ChunkedCacheConfig,
  ) {
    this.method = method
    this.url_template = url_template
    this.headers_template = headers_template;
    this.chunked_cache_config = chunked_cache_config;
  }
}

export class ChunkedCacheConfig {
  chunk_size: number;
  cache_size?: number;
  constructor(
    chunk_size: number,
    cache_size?: number
  ) {
    this.chunk_size = chunk_size
    this.cache_size = cache_size
  }
}
