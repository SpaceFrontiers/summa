import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import type { IndexPayload } from "summa-wasm/web-index-service";
import type { NetworkConfig } from "summa-wasm/network-config";
import Dexie from "dexie";
import { toRaw } from "vue";

class SummaDatabase extends Dexie {
  index_configs!: Dexie.Table<IIndexConfig, string>;

  constructor() {
    super("SummaDatabase");
    this.version(2).stores({
      index_configs: "index_payload.name,is_enabled",
    });
    this.index_configs.mapToClass(IndexConfig);
  }
}

interface IIndexConfig {
  is_enabled: boolean;
  is_warm_up: boolean;
  index_payload: IndexPayload;
  ipns_path: IPFSPath;
  network_config: NetworkConfig;
}

export class IndexConfig implements IIndexConfig {
  is_enabled: boolean;
  is_warm_up: boolean;
  index_payload: IndexPayload;
  ipns_path: IPFSPath;
  network_config: NetworkConfig;

  constructor(
    is_enabled: boolean,
    is_warm_up: boolean,
    index_payload: IndexPayload,
    ipns_path: IPFSPath,
    network_config: NetworkConfig
  ) {
    this.is_enabled = is_enabled;
    this.is_warm_up = is_warm_up;
    this.index_payload = index_payload;
    this.ipns_path = ipns_path;
    this.network_config = network_config;
  }

  get_pin_command(): string {
    return "ipfs name resolve " + this.ipns_path + " | ipfs pin add";
  }

  save() {
    return db.transaction("rw", db.index_configs, () => {
      return db.index_configs.put(toRaw(this));
    });
  }
}

export const db = new SummaDatabase();
