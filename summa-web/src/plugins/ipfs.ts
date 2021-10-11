import axios from "axios";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";

function detect_ipfs_url() {
  let ipfs_url = window.location.hostname;
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
export const ipfs_url = detect_ipfs_url();

async function resolve_file(
  ipfs_name: IPFSPath,
  file_name: string
): Promise<[string, number]> {
  if (file_name.endsWith(".json")) {
    const response = await axios.get(`${ipfs_url}${ipfs_name}/${file_name}`, {
      responseType: "arraybuffer",
    });
    return [file_name, response.data.byteLength as number];
  } else {
    const file_response = await axios.head(
      `${ipfs_url}${ipfs_name}/${file_name}`
    );
    return [
      file_name,
      parseInt(file_response.headers["content-length"]) as number,
    ];
  }
}

export class IPFSGatewayClient {
  url: string;

  constructor(url: string) {
    this.url = url;
  }

  async resolve(ipns_name: IPFSPath): Promise<string> {
    const response = await axios.get(this.url + ipns_name);
    return "/ipfs/" + response.headers["x-ipfs-roots"];
  }

  async ls(ipfs_name: IPFSPath) {
    const response = await axios.get(`${ipfs_url}${ipfs_name}`);
    const parser = new DOMParser();
    const html_doc = parser.parseFromString(response.data, "text/html");
    const promises: Promise<[string, number]>[] = Array.from(
      html_doc.querySelectorAll("tr td:nth-child(2) a")
    )
      .map((e) => e.innerHTML)
      .filter((file_name) => file_name !== "..")
      .map((file_name) => resolve_file(ipfs_name, file_name));
    return new Map(await Promise.all(promises));
  }
  async cat(ipfs_name: IPFSPath) {
    const response = await axios.get(`${ipfs_url}${ipfs_name}`, {
      responseType: "arraybuffer",
    });
    return new Uint8Array(response.data);
  }
}
export const ipfs = new IPFSGatewayClient(ipfs_url);
