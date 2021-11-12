import type { NetworkConfig, StatusCallback } from "summa-wasm";
import Dexie from "dexie";
import { toRaw } from "vue";

class SummaDatabase extends Dexie {
  index_configs!: Dexie.Table<IIndexConfig, string>;

  constructor() {
    super("SummaDatabase");
    this.version(5).stores({
      index_configs: "index_payload.name,is_enabled",
    });
    this.index_configs.mapToClass(IndexConfig);
  }
}

export interface IIndexSeed {
  retrieve_network_config(
    status_callback: StatusCallback
  ): Promise<NetworkConfig>;
  get_pin_command(): string;
}

interface IIndexConfig {
  is_enabled: boolean;
  is_warm_up: boolean;
  index_payload: Object;
  index_seed: Object;
  network_config: NetworkConfig;
}

export class IndexConfig implements IIndexConfig {
  is_enabled: boolean;
  is_warm_up: boolean;
  index_payload: Object;
  index_seed: Object;
  network_config: NetworkConfig;

  constructor(
    is_enabled: boolean,
    is_warm_up: boolean,
    index_payload: Object,
    index_seed: Object,
    network_config: NetworkConfig
  ) {
    this.is_enabled = is_enabled;
    this.is_warm_up = is_warm_up;
    this.index_payload = index_payload;
    this.index_seed = index_seed;
    this.network_config = network_config;
  }

  save() {
    return db.transaction("rw", db.index_configs, () => {
      return db.index_configs.put(toRaw(this));
    });
  }
}

export const db = new SummaDatabase();
