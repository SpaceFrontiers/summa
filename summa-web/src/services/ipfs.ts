import axios from "axios";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import {support_subdomains} from "@/options";

function detect_ipfs_url(hostname: string) {
  let ipfs_url = hostname;
  if (
    process.env.NODE_ENV === "development" ||
    window.location.port === "4173"
  ) {
    return "http://localhost:8080";
  }
  const hostname_parts = window.location.hostname.split(".");
  if (hostname_parts[-1] === "localhost") {
    ipfs_url = "http://localhost";
  } else if (window.location.hostname === "ipfs.io") {
    ipfs_url = "https://ipfs.io";
  } else {
    const ipfs_domain_index = hostname_parts.findIndex(
      (el) => el === "ipfs" || el === "ipns"
    );
    if (ipfs_domain_index !== undefined) {
      ipfs_url = `${window.location.protocol}//${hostname_parts
        .slice(ipfs_domain_index + 1)
        .join(".")}`;
    }
  }
  if (
    window.location.port !== undefined &&
    window.location.port !== "" &&
    window.location.port !== "80"
  ) {
    ipfs_url = `${ipfs_url}:${window.location.port}`;
  }
  return ipfs_url;
}
export const ipfs_url = detect_ipfs_url(window.location.hostname);
const parsed_url = new URL(ipfs_url);
let hostname = parsed_url.hostname;
if (parsed_url.port !== "") {
  hostname += ":" + parsed_url.port;
}
export const ipfs_hostname = hostname;
export const ipfs_http_protocol = parsed_url.protocol;

function get_ipfs_url(options: {
  ipfs_hash?: string;
  ipns_hash?: string;
  file_name?: string;
  subdomain: boolean;
}) {
  const { ipfs_hash, ipns_hash, file_name, subdomain } = options;
  let url = "";
  if (subdomain) {
    url = ipfs_http_protocol + "//";
    if (ipfs_hash) {
      url += ipfs_hash + ".ipfs.";
    } else if (ipns_hash) {
      url += ipns_hash + ".ipns.";
    } else {
      throw new Error("No IPFS or IPNS hashes");
    }
    url += ipfs_hostname;
  } else {
    url = ipfs_url + "/";
    if (ipfs_hash) {
      url += "ipfs/" + ipfs_hash;
    } else if (ipns_hash) {
      url += "ipns/" + ipns_hash;
    } else {
      throw new Error("No IPFS or IPNS hashes");
    }
  }
  url += "/";
  if (file_name !== undefined) {
    return `${url}${file_name}`;
  }
  return url;
}

async function resolve_file(
  ipfs_hash: string,
  file_name: string
): Promise<[string, number]> {
  const file_url = get_ipfs_url({
    ipfs_hash: ipfs_hash,
    file_name: file_name,
    subdomain: support_subdomains,
  });
  if (file_name.endsWith(".json")) {
    const response = await axios.get(file_url, {
      responseType: "arraybuffer",
    });
    return [file_name, response.data.byteLength as number];
  } else {
    const file_response = await axios.head(file_url);
    return [
      file_name,
      parseInt(file_response.headers["content-length"] || "0") as number,
    ];
  }
}

export class IPFSGatewayClient {
  async resolve(ipns_hash: string): Promise<string> {
    const response = await axios.get(
      get_ipfs_url({ ipns_hash: ipns_hash, subdomain: support_subdomains })
    );
    return "/ipfs/" + response.headers["x-ipfs-roots"];
  }

  async ls(ipfs_hash: string) {
    const response = await axios.get(
      get_ipfs_url({ ipfs_hash: ipfs_hash, subdomain: support_subdomains })
    );
    const parser = new DOMParser();
    const html_doc = parser.parseFromString(response.data, "text/html");
    const promises: Promise<[string, number]>[] = Array.from(
      html_doc.querySelectorAll("tr td:nth-child(2) a")
    )
      .map((e) => e.innerHTML)
      .filter((file_name) => file_name !== "..")
      .map((file_name) => resolve_file(ipfs_hash, file_name));
    return new Map(await Promise.all(promises));
  }
  async cat(ipfs_name: IPFSPath) {
    const response = await axios.get(`${ipfs_url}${ipfs_name}`, {
      responseType: "arraybuffer",
    });
    return new Uint8Array(response.data);
  }
}
export const ipfs = new IPFSGatewayClient();
