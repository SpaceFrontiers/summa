import type { Remote } from "comlink";
import * as Comlink from "comlink";
import type { IndexQuery, StatusCallback, WebIndexService } from "summa-wasm";
import { ChunkedCacheConfig, NetworkConfig } from "summa-wasm";
import { ref, toRaw } from "vue";
import { db, type IIndexSeed, IndexConfig } from "@/database";
import {
  ipfs_hostname,
  ipfs_http_protocol,
  ipfs_url,
  is_eth_hostname,
  is_supporting_subsubdomains,
} from "@/options";
import axios from "axios";

export class IpfsDatabaseSeed implements IIndexSeed {
  ipfs_path: string;

  constructor(ipfs_path: string) {
    this.ipfs_path = ipfs_path;
  }

  get_pin_command(): string {
    if (this.ipfs_path.startsWith("/ipfs")) {
      return `ipfs pin add ${this.ipfs_path}`;
    } else {
      return `ipfs name resolve ${this.ipfs_path} | ipfs pin add`;
    }
  }

  async retrieve_network_config(
    status_callback: StatusCallback
  ): Promise<NetworkConfig> {
    status_callback("status", `resolving ${this.ipfs_path}...`);
    const response = await axios.get(ipfs_url + this.ipfs_path);
    let ipfs_hash = response.headers["x-ipfs-roots"];
    if (
      ipfs_hash === undefined &&
      response.headers["content-type"] === "text/html"
    ) {
      const parser = new DOMParser();
      const htmlDoc = parser.parseFromString(response.data, "text/html");
      ipfs_hash = htmlDoc
        .getElementsByClassName("ipfs-hash")[0]
        .textContent!.trim();
    }
    console.debug("Detected IPFS Hash", ipfs_hash);
    try {
      // ToDo: Create separate check function
      await axios.get(`${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/meta.json`);
      return new NetworkConfig(
        "GET",
        `${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/{file_name}`,
        [{ name: "range", value: "bytes={start}-{end}" }],
        new ChunkedCacheConfig(16 * 1024, 128 * 1024 * 1024)
      );
    } catch {
      return new NetworkConfig(
        "GET",
        `${ipfs_http_protocol}//${ipfs_hostname}/ipfs/${ipfs_hash}/{file_name}`,
        [{ name: "range", value: "bytes={start}-{end}" }],
        new ChunkedCacheConfig(16 * 1024, 128 * 1024 * 1024)
      );
    }
  }
}

class EthSubdomainDatabaseSeed implements IIndexSeed {
  subdomain: string;

  constructor(subdomain: string) {
    this.subdomain = subdomain;
  }

  get_pin_command(): string {
    return "/ipns/" + this.subdomain + ".summa-t.eth";
  }

  async retrieve_network_config(
    status_callback: StatusCallback
  ): Promise<NetworkConfig> {
    status_callback("status", `resolving files...`);
    const url = `${ipfs_http_protocol}//${this.subdomain}.${ipfs_hostname}/`;
    return new NetworkConfig(
      "GET",
      `${url}{file_name}`,
      [{ name: "range", value: "bytes={start}-{end}" }],
      new ChunkedCacheConfig(16 * 1024, 128 * 1024 * 1024)
    );
  }
}

async function get_startup_configs() {
  if (!(await is_supporting_subsubdomains()) && is_eth_hostname) {
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
      seed: new IpfsDatabaseSeed("/ipfs/bafykbzacebftn62im7khi24gstiqc6j4e2bxpodttx5tnveecr3brc5uhqwvw/"),
      is_enabled: true,
    },
    {
      seed: new IpfsDatabaseSeed("/ipfs/bafykbzacebghljvglld3ycd4jglv45sqb3rfqgaygj7ansgejd47skn3xuawm/"),
      is_enabled: false,
    },
  ];
}

export class SearchService {
  status_callback: StatusCallback;
  web_index_service_worker: Remote<WebIndexService>;

  constructor() {
    this.status_callback = (type: string, message: string) =>
      console.log(type, message);
    this.web_index_service_worker = Comlink.wrap<WebIndexService>(
      new Worker(
        new URL(
          "../../node_modules/summa-wasm/dist/worker.js",
          import.meta.url
        ),
        { type: "module" }
      )
    );
  }
  async setup(options: { num_threads: number }) {
    try {
      await this.web_index_service_worker.setup(
        new URL(
          "../../node_modules/summa-wasm/dist/index_bg.wasm",
          import.meta.url
        ).href,
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
      const network_config = toRaw(index_config.network_config);
      // ToDo: remove soon
      network_config.chunked_cache_config = new ChunkedCacheConfig(
        16 * 1024,
        128 * 1024 * 1024
      );
      await this.web_index_service_worker.add({ remote: network_config });
    }
  }
  async add_index(startup_config: {
    seed: IIndexSeed;
    is_enabled: boolean;
  }): Promise<Object> {
    const network_config = await startup_config.seed.retrieve_network_config(
      this.status_callback
    );
    const index_payload = await this.web_index_service_worker.add({
      remote: network_config,
    });
    const index_config = new IndexConfig(
      startup_config.is_enabled,
      false,
      index_payload,
      startup_config.seed,
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
