import { summa } from "./proto";

export interface IIndexSeed {
  retrieve_remote_engine_config(): Promise<summa.proto.RemoteEngineConfig>;
}

export class LocalDatabaseSeed implements IIndexSeed {
  ipfs_path: string;
  cache_config?: summa.proto.CacheConfig;

  constructor(ipfs_path: string, cache_config?: summa.proto.CacheConfig) {
    if (!ipfs_path.endsWith("/")) {
      ipfs_path += "/";
    }
    if (!ipfs_path.startsWith("/")) {
      ipfs_path = "/" + ipfs_path;
    }
    this.ipfs_path = ipfs_path;
    this.cache_config = cache_config;
  }

  async retrieve_remote_engine_config(): Promise<summa.proto.RemoteEngineConfig> {
    return summa.proto.RemoteEngineConfig.create({
      method: "GET",
      url_template: `${this.ipfs_path}{file_name}`,
      headers_template: { range: "bytes={start}-{end}" },
      cache_config: this.cache_config,
    });
  }
}
