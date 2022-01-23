import {ChunkedCacheConfig, RemoteEngineConfig} from "./configs";

export interface IIndexSeed {
  retrieve_remote_engine_config(): Promise<RemoteEngineConfig>;
}

export class LocalDatabaseSeed implements IIndexSeed {
  ipfs_path: string;
  chunked_cache_config: ChunkedCacheConfig;

  constructor(ipfs_path: string, chunked_cache_config: ChunkedCacheConfig) {
    if (!ipfs_path.endsWith("/")) {
      ipfs_path += "/";
    }
    this.ipfs_path = ipfs_path;
    this.chunked_cache_config = chunked_cache_config;
  }

  async retrieve_remote_engine_config(): Promise<RemoteEngineConfig> {
    return new RemoteEngineConfig(
      "GET",
      `${this.ipfs_path}{file_name}`,
      new Map([["range", "bytes={start}-{end}"]]),
      this.chunked_cache_config,
    );
  }
}