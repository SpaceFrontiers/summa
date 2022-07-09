import { summa } from "./proto";
import Dexie from "dexie";

export class MetaDb extends Dexie {
  index_configs!: Dexie.Table<IIndexConfig, string>;
  target_version!: number;

  constructor(name: string, version: number) {
    super(name);
    this.target_version = version;
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

  public open() {
    if (this.isOpen()) return super.open();
    return Dexie.Promise.resolve()
      .then(() => Dexie.exists(this.name))
      .then((exists) => {
        if (!exists) {
          // no need to check database version since it doesn't exist
          return;
        }

        // Open separate instance of dexie to get current database version
        return new Dexie(this.name).open()
          .then(async db => {
            if (db.verno >= this.target_version) {
              // database up to date (or newer)
              return db.close();
            }

            console.log(`Database schema out of date, resetting all data. (currentVersion: ${db.verno}, expectedVersion: ${this.target_version})`);
            await db.delete();

            // ensure the deletion was successful
            const exists = await Dexie.exists(this.name);
            if (exists) {
              throw new Error('Failed to remove mock backend database.');
            }
          })
      })
      .then(() => super.open());
  }
}

interface IIndexConfig {
  index_name: string;
  index_seed: Object;
  index_properties: Object;
  remote_engine_config: summa.proto.RemoteEngineConfig;
  query_parser_config: summa.proto.QueryParserConfig;
}

export class IndexConfig implements IIndexConfig {
  index_name: string;
  description: string;
  created_at: number;
  index_seed: Object;
  remote_engine_config: summa.proto.RemoteEngineConfig;
  query_parser_config: summa.proto.QueryParserConfig;
  index_properties: Object;

  constructor(
    index_name: string,
    description: string,
    created_at: number,
    index_seed: Object,
    remote_engine_config: summa.proto.RemoteEngineConfig,
    query_parser_config: summa.proto.QueryParserConfig,
    index_properties: Object,
  ) {
    this.index_name = index_name;
    this.description = description;
    this.created_at = created_at;
    this.index_seed = index_seed;
    this.remote_engine_config = remote_engine_config;
    this.query_parser_config = query_parser_config;
    this.index_properties = index_properties;
  }
}
