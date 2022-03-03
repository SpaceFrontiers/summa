import {CacheConfig, RemoteEngineConfig} from "./configs";
import {get_ipfs_hostname, get_ipfs_url} from "./utils";
import axios from "axios";

export interface IIndexSeed {
  retrieve_remote_engine_config(): Promise<RemoteEngineConfig>;
}

export class LocalDatabaseSeed implements IIndexSeed {
  ipfs_path: string;
  cache_config?: CacheConfig;
  timeout_ms?: number;

  constructor(ipfs_path: string, cache_config?: CacheConfig, timeout_ms?: number) {
    if (!ipfs_path.endsWith("/")) {
      ipfs_path += "/";
    }
    this.ipfs_path = ipfs_path;
    this.cache_config = cache_config;
    this.timeout_ms = timeout_ms;
  }

  async retrieve_remote_engine_config(): Promise<RemoteEngineConfig> {
    return new RemoteEngineConfig(
      "GET",
      `${this.ipfs_path}{file_name}`,
      new Map([["range", "bytes={start}-{end}"]]),
      this.cache_config,
      this.timeout_ms,
    );
  }
}

export class IpfsDatabaseSeed implements IIndexSeed {
  ipfs_path: string;
  cache_config: CacheConfig;
  ipfs_url?: string;
  timeout_ms?: number;

  constructor(ipfs_path: string, cache_config: CacheConfig, ipfs_url?: string, timeout_ms?: number) {
    this.ipfs_path = ipfs_path;
    this.cache_config = cache_config;
    this.ipfs_url = ipfs_url;
    this.timeout_ms = timeout_ms;
  }

  async retrieve_remote_engine_config(): Promise<RemoteEngineConfig> {
    const ipfs_url = this.ipfs_url || get_ipfs_url();
    const { ipfs_hostname, ipfs_http_protocol } = get_ipfs_hostname(ipfs_url)
    const response = await axios.get(ipfs_url + this.ipfs_path);
    let ipfs_hash = response.headers["x-ipfs-roots"];
    if (
      ipfs_hash === undefined &&
      response.headers["content-type"] === "text/html"
    ) {
      const parser = new DOMParser();
      const htmlDoc = parser.parseFromString(response.data, "text/html");
      if (htmlDoc.getElementsByClassName("ipfs-hash").length > 0) {
        // Kubo
        ipfs_hash = htmlDoc
          .getElementsByClassName("ipfs-hash")[0]
          .textContent!.trim();
      } else {
        // Iroh
        ipfs_hash = htmlDoc
          .getElementsByTagName("title")[0]
          .textContent!.replace("/ipfs/", "")
          .trim();
        if (ipfs_hash.endsWith("/")) {
          ipfs_hash = ipfs_hash.substring(0, ipfs_hash.length - 1);
        }
      }
    }
    try {
      // ToDo: Create separate check function
      await axios.get(
        `${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/meta.json`
      );
      return new RemoteEngineConfig(
        "GET",
        `${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/{file_name}`,
        new Map([["range", "bytes={start}-{end}"]]),
        this.cache_config,
        this.timeout_ms,
      );
    } catch {
      return new RemoteEngineConfig(
        "GET",
        `${ipfs_http_protocol}//${ipfs_hostname}/ipfs/${ipfs_hash}/{file_name}`,
        new Map([["range", "bytes={start}-{end}"]]),
        this.cache_config,
        this.timeout_ms,
      );
    }
  }
}
