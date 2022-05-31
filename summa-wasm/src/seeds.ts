import { summa } from "./proto";
import { get_ipfs_hostname, get_ipfs_url } from "./utils";
import axios from "axios";

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

export class IpfsDatabaseSeed implements IIndexSeed {
  ipfs_path: string;
  cache_config: summa.proto.CacheConfig;
  ipfs_url?: string;

  constructor(ipfs_path: string, cache_config: summa.proto.CacheConfig, ipfs_url?: string) {
    this.ipfs_path = ipfs_path;
    this.cache_config = cache_config;
    this.ipfs_url = ipfs_url;
  }

  async retrieve_remote_engine_config(): Promise<summa.proto.RemoteEngineConfig> {
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
      return summa.proto.RemoteEngineConfig.create({
        method: "GET",
        url_template: `${ipfs_http_protocol}//${ipfs_hash}.ipfs.${ipfs_hostname}/{file_name}`,
        headers_template: { range: "bytes={start}-{end}" },
        cache_config: this.cache_config,
      });
    } catch {
      return summa.proto.RemoteEngineConfig.create({
        method: "GET",
        url_template: `${ipfs_http_protocol}//${ipfs_hostname}/ipfs/${ipfs_hash}/{file_name}`,
        headers_template: { range: "bytes={start}-{end}" },
        cache_config: this.cache_config,
      });
    }
  }
}
