import { detect } from "detect-browser";

export const browser = detect();
export const num_threads =
  browser && (browser.name === "safari" || browser.name === "ios") ? 0 : 16;
export const support_subdomains =
  window.location.hostname !== "lib.kropotkin.rocks" &&
  window.location.hostname !== "cloudflare-ipfs.com";

console.debug("browser", browser);
console.debug("num_threads", num_threads);
console.debug("support_subdomains", support_subdomains);
