import * as Comlink from "comlink";
import type {
  IndexPayload,
  IndexQuery,
  WebIndexService as WebIndexServiceWasm,
} from "summa-wasm/web-index-service";
import { db, type IIndexSeed, IndexConfig } from "@/database";
import { ref, toRaw } from "vue";
import type { Remote } from "comlink";
import { ipfs_hostname, ipfs_http_protocol } from "@/options";
import { NetworkConfig } from "summa-wasm/network-config";
import { is_eth_hostname, is_supporting_subdomains } from "@/options";
import { get_ipfs_url, ipfs } from "@/services/ipfs";

class IpnsDatabaseSeed implements IIndexSeed {
  ipns_path: string;

  constructor(ipns_path: string) {
    this.ipns_path = ipns_path;
  }
  get_ipns(): string {
    return this.ipns_path;
  }
  async retrieve_network_config(
    status_callback: StatusCallback
  ): Promise<NetworkConfig> {
    status_callback("status", `resolving ${this.ipns_path}...`);
    const ipfs_path = await ipfs.resolve(
      (this.ipns_path as string).split("/")[2]
    );
    const ipfs_hash = ipfs_path.split("/")[2] as string;
    status_callback("status", `resolving files...`);
    const files = await ipfs.ls(get_ipfs_url({ ipfs_hash }));
    return new NetworkConfig(
      "GET",
      `${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/{file_name}`,
      [{ name: "range", value: "bytes={start}-{end}" }],
      files
    );
  }
}

class EthSubdomainDatabaseSeed implements IIndexSeed {
  subdomain: string;

  constructor(subdomain: string) {
    this.subdomain = subdomain;
  }
  get_ipns(): string {
    return "/ipns/" + this.subdomain + ".summa-t.eth";
  }
  async retrieve_network_config(
    status_callback: StatusCallback
  ): Promise<NetworkConfig> {
    status_callback("status", `resolving files...`);
    const url = `${ipfs_http_protocol}//${this.subdomain}.${ipfs_hostname}/`;
    const files = await ipfs.ls(url);
    return new NetworkConfig(
      "GET",
      `${url}{file_name}`,
      [{ name: "range", value: "bytes={start}-{end}" }],
      files
    );
  }
}

async function get_startup_configs() {
  if (!(await is_supporting_subdomains()) && is_eth_hostname) {
    return [
      {
        seed: new EthSubdomainDatabaseSeed("nexus-books"),
        is_enabled: true,
      },
      {
        seed: new EthSubdomainDatabaseSeed("nexus-media"),
        is_enabled: false,
      },
    ];
  }
  return [
    {
      seed: new IpnsDatabaseSeed("/ipns/nexus-books.summa-t.eth/"),
      is_enabled: true,
    },
    {
      seed: new IpnsDatabaseSeed("/ipns/nexus-media.summa-t.eth/"),
      is_enabled: false,
    },
  ];
}

export type StatusCallback = (type: string, message: string) => void;
export class WebIndexService {
  status_callback: StatusCallback;
  web_index_service_worker: Remote<WebIndexServiceWasm>;

  constructor() {
    this.status_callback = (type: string, message: string) =>
      console.log(type, message);
    this.web_index_service_worker = Comlink.wrap<WebIndexServiceWasm>(
      new Worker(
        new URL(
          "../../node_modules/summa-wasm/web-index-service.ts",
          import.meta.url
        ),
        { type: "module" }
      )
    );
  }
  async setup(options: { num_threads: number }) {
    try {
      await this.web_index_service_worker.setup(
        options.num_threads,
        Comlink.proxy(this.status_callback)
      );
      await this.load_from_store();
    } catch (e) {
      console.error("Dropping stored data due to error: ", e);
      await db.index_configs.clear();
      throw e;
    }
    try {
      if (await this.is_empty()) {
        await this.install_defaults();
      }
    } catch (e) {
      console.error(e);
      return false;
    }
    return true;
  }
  async load_from_store() {
    for (const index_config of await db.index_configs.toArray()) {
      await this.web_index_service_worker.add(
        toRaw(index_config.network_config)
      );
    }
  }
  async add_index(startup_config: {
    seed: IIndexSeed;
    is_enabled: boolean;
  }): Promise<IndexPayload> {
    const network_config = await startup_config.seed.retrieve_network_config(
      this.status_callback
    );
    const index_payload = await this.web_index_service_worker.add(
      network_config
    );
    const index_config = new IndexConfig(
      startup_config.is_enabled,
      false,
      index_payload,
      startup_config.seed.get_ipns(),
      network_config
    );
    await index_config.save();
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    return index_payload;
  }
  async delete_index(index_name: string) {
    await this.web_index_service_worker.delete(index_name);
    await db.index_configs.delete(index_name);
  }
  async search(index_queries: IndexQuery[]) {
    const response = await this.web_index_service_worker.search(index_queries);
    cache_metrics.value = await this.web_index_service_worker.cache_metrics();
    return response;
  }
  async is_empty() {
    return (await db.index_configs.count()) == 0;
  }
  async install_defaults() {
    const startup_configs = await get_startup_configs();
    return await Promise.all(
      startup_configs.map((startup_config) => this.add_index(startup_config))
    );
  }
}

export const cache_metrics = ref({
  requests: 0,
  bytes_received: 0,
});
