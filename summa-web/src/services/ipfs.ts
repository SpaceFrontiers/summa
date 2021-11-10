import axios from "axios";
import { ipfs_hostname, ipfs_http_protocol } from "@/options";

export class IPFSGatewayClient {
  async resolve(ipns_name: string): Promise<string> {
    const response = await axios.get(get_ipfs_url({ ipns_name: ipns_name }));
    return "/ipfs/" + response.headers["x-ipfs-roots"];
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
