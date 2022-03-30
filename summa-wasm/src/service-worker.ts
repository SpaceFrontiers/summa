/// <reference lib="webworker" />
export default null
declare let self: ServiceWorkerGlobalScope

function delay(t: any) {
    return new Promise(resolve => setTimeout(resolve, t));
}

async function fetch_with_retries(url: string, options: any, retries: number = 5, delay: number = 1000): Promise<Response> {
  try {
    const res = await fetch(url, options);
    if (!res.ok && retries > 0) {
      throw res;
    }
    return res;
  } catch (error: any) {
    console.debug("retry failed", error);
    if (retries === 0) {
      throw error;
    }
    if (error.status == 503 || error.status == 502 || error.name == 'AbortError') {
      await new Promise(resolve => setTimeout(resolve, delay));
      return fetch_with_retries(url, options, retries - 1, delay * 2);
    } else {
      return fetch_with_retries(url, options, retries - 1, delay);
    }
  }
}

function set_same_origin_headers(headers: Headers) {
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  return headers;
}

async function handle_request(event: FetchEvent) {
  const request = event.request
  let filename = request.url;
  let url = request.url;
  const range_header = request.headers.get("range");
  if (range_header !== null) {
    let range_end = '';
    // @ts-ignore
    const [_, range_start, end] = /^bytes=(\d+)-(\d+)?$/g.exec(range_header);
    if (end) {
      range_end = end;
    }
    url += "?r=" + range_start + "-" + range_end;
  }

  let is_immutable_file = filename.endsWith(".fast") ||
      filename.endsWith(".term") ||
      filename.endsWith(".pos") ||
      filename.endsWith(".store") ||
      filename.endsWith(".fieldnorm") ||
      filename.endsWith(".idx") ||
      filename.endsWith(".del") ||
      filename.endsWith(".wasm") ||
      filename.endsWith(".bin") ||
      (filename.endsWith(".json") && !filename.endsWith("meta.json")) ||
      event.request.destination === "image" ||
      event.request.destination === "font" ||
      event.request.destination === "style" ||
      (event.request.destination === "script" && !request.url.startsWith("chrome-extension"));

  let caching_enabled = is_immutable_file && request.method === "GET";

  const cache = await caches.open("cache_v2");
  let response = undefined;
  if (caching_enabled) {
      response = await cache.match(url);
  }
  if (response === undefined) {
    response = await fetch_with_retries(
      url,
      {
        method: request.method,
        headers: request.headers,
      },
      5,
        1000,
    );
    response = new Response(response.body, {
      headers: set_same_origin_headers(new Headers(response.headers)),
      status: response.status === 206 ? 200 : response.status
    })
    if (caching_enabled && response.ok) {
      cache.put(url, response.clone());
    }
  }
  return response;
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) =>
    event.waitUntil((async () => {
    if (self.registration.navigationPreload) {
      // Disable navigation preloads!
      await self.registration.navigationPreload.disable();
    }
    await self.clients.claim();
  })())
);

self.addEventListener("message", (ev) => {
  if (ev.data && ev.data.type === "deregister") {
    self.registration
      .unregister()
      .then(() => {
        return self.clients.matchAll();
      })
      .then((clients) => {
        // @ts-ignore
        clients.forEach((client) => client.navigate(client.url));
      });
  }
});
self.addEventListener("fetch", (event) => {
  if (
    event.request.cache === "only-if-cached" &&
    event.request.mode !== "same-origin"
  ) {
    return;
  }
  event.respondWith(handle_request(event));
});