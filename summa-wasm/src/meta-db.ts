import type { RemoteEngineConfig } from "./configs";
import Dexie from "dexie";

export class MetaDb extends Dexie {
  index_configs!: Dexie.Table<IIndexConfig, string>;

  constructor(name: string, version: number) {
    super(name);
    this.version(version).stores({
      index_configs: "index_name",
    });
    this.index_configs.mapToClass(IndexConfig);
  }

  save(item: IIndexConfig) {
    return this.transaction("rw", this.index_configs, () => {
      return this.index_configs.put(item);
    });
  }
}

interface IIndexConfig {
  index_name: string;
  index_seed: Object;
  index_properties: Object;
  remote_engine_config: RemoteEngineConfig;
}

export class IndexConfig implements IIndexConfig {
  index_name: string;
  description: string;
  created_at: number;
  index_seed: Object;
  index_properties: Object;
  remote_engine_config: RemoteEngineConfig;

  constructor(
    index_name: string,
    description: string,
    created_at: number,
    index_seed: Object,
    index_properties: Object,
    remote_engine_config: RemoteEngineConfig
  ) {
    this.index_name = index_name;
    this.description = description;
    this.created_at = created_at;
    this.index_seed = index_seed;
    this.index_properties = index_properties;
    this.remote_engine_config = remote_engine_config;
  }
}
