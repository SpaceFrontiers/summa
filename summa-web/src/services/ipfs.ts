import axios from "axios";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import { ipfs_hostname, ipfs_http_protocol, ipfs_url } from "@/options";

export class IPFSGatewayClient {
  async resolve(ipns_name: string): Promise<string> {
    const response = await axios.get(get_ipfs_url({ ipns_name: ipns_name }));
    return "/ipfs/" + response.headers["x-ipfs-roots"];
  }

  async ls(url: string) {
    const response = await axios.get(url);
    const parser = new DOMParser();
    const html_doc = parser.parseFromString(response.data, "text/html");
    const promises: Promise<[string, number]>[] = Array.from(
      html_doc.querySelectorAll("tr td:nth-child(2) a")
    )
      .map((e) => e.innerHTML)
      .filter((file_name) => file_name !== "..")
      .map((file_name) => resolve_file(url, file_name));
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

export function get_ipfs_url(options: {
  ipfs_hash?: string;
  ipns_name?: string;
  file_name?: string;
}) {
  const { ipfs_hash, ipns_name, file_name } =
    options;
  let url = ipfs_http_protocol + "//";
  if (ipfs_hash) {
    url += ipfs_hash + ".ipfs.";
  } else if (ipns_name) {
    url += ipns_name + ".ipns.";
  } else {
    throw new Error("No IPFS or IPNS hashes");
  }
  url += ipfs_hostname;
  url += "/";
  if (file_name !== undefined) {
    return `${url}${file_name}`;
  }
  return url;
}

async function resolve_file(
  url: string,
  file_name: string
): Promise<[string, number]> {
  const file_url = url + file_name;
  if (file_name.endsWith(".json")) {
    const response = await axios.get(url, {
      responseType: "arraybuffer",
    });
    return [file_url, response.data.byteLength as number];
  } else {
    const file_response = await axios.head(file_url);
    return [
      file_name,
      parseInt(file_response.headers["content-length"] || "0") as number,
    ];
  }
}
