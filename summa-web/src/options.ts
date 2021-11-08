import { detect } from "detect-browser";
import axios from "axios";

export const is_development = process.env.NODE_ENV === "development" || window.location.port === "4173";
export const browser = detect();
export const num_threads =
  browser && (browser.name === "safari" || browser.name === "ios") ? 0 : 16;
export const hostname = window.location.hostname + (window.location.port == "" ? "" : ":" + window.location.port);
export const ipfs_url = detect_ipfs_url(window.location.hostname);
export const { ipfs_hostname, ipfs_http_protocol } = get_ipfs_hostname(ipfs_url);
export const is_eth_hostname =
  hostname.endsWith("eth.link") || hostname.endsWith("eth.limo");

function get_ipfs_hostname(ipfs_url: string) {
  const parsed_url = new URL(ipfs_url);
  let ipfs_hostname = parsed_url.hostname;
  if (parsed_url.port !== "") {
    ipfs_hostname += ":" + parsed_url.port;
  }
  return { ipfs_hostname, ipfs_http_protocol: parsed_url.protocol };
}

function detect_ipfs_url(hostname: string) {
  let ipfs_url = hostname;
  if (is_development) {
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

export async function is_supporting_subdomains() {
  try {
    await axios.get(
      ipfs_http_protocol + "//nexus-books.summa-t.eth.ipns." + ipfs_hostname
    );
    return true;
  } catch (e) {
    return false;
  }
}

console.debug("browser", browser);
console.debug("num_threads", num_threads);
console.debug("ipfs_url", ipfs_url);
