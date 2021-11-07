import { detect } from "detect-browser";

export const browser = detect();
export const num_threads =
  browser && (browser.name === "safari" || browser.name === "ios") ? 0 : 16;

console.debug("browser", browser);
console.debug("num_threads", num_threads);
