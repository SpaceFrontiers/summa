export function get_ipfs_hostname() {
  let ipfs_url: string;
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
    } else {
      ipfs_url = "https://ipfs.io";
    }
  }
  if (
    window.location.port !== undefined &&
    window.location.port !== "" &&
    window.location.port !== "80"
  ) {
    if (window.location.port !== "5173") {
      ipfs_url = `${ipfs_url}:${window.location.port}`;
    } else {
      ipfs_url = `${ipfs_url}:8080`;
    }
  }
  return ipfs_url;
}