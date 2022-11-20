export function get_ipfs_hostname() {
  let hostname: string = window.location.hostname;
  let protocol: string = window.location.protocol;
  const hostname_parts = hostname.split(".");
  if (hostname_parts[hostname_parts.length - 1] === "localhost") {
    hostname = "localhost";
    protocol = "http:";
  } else if (hostname === "ipfs.io") {
    hostname = "ipfs.io";
    protocol = "https:";
  } else {
    const ipfs_domain_index = hostname_parts.findIndex(
      (el) => el === "ipfs" || el === "ipns"
    );
    if (ipfs_domain_index !== -1) {
      hostname = hostname_parts
        .slice(ipfs_domain_index + 1)
        .join(".");
    } else {
      hostname = "ipfs.io";
      protocol = "https:";
    }
  }
  if (
    window.location.port !== undefined &&
    window.location.port !== "" &&
    window.location.port !== "80"
  ) {
    if (window.location.port !== "5173") {
      hostname = `${hostname}:${window.location.port}`;
    } else {
      hostname = `${hostname}:8080`;
    }
  }
  return { ipfs_hostname: hostname, ipfs_protocol: protocol };
}