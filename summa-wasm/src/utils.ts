export function get_ipfs_hostname(ipfs_url?: string) {
  ipfs_url = ipfs_url || get_ipfs_url();
  const parsed_url = new URL(ipfs_url);
  let ipfs_hostname = parsed_url.hostname;
  if (parsed_url.port !== "") {
    if (parsed_url.port !== "5173") {
      ipfs_hostname += ":" + parsed_url.port;
    } else {
      ipfs_hostname += ":8080";
    }
  }
  return { ipfs_hostname, ipfs_http_protocol: parsed_url.protocol };
}

export function get_ipfs_url(hostname?: string) {
  let ipfs_url = hostname || window.location.hostname;
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