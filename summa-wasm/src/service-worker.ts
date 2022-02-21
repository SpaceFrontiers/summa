/// <reference lib="webworker" />
export default null
declare let self: ServiceWorkerGlobalScope

function fetch_with_retries(url: string, options: any, retries: number): any {
  return fetch(url, options)
    .then((res) => {
      if (!res.ok && retries > 0) {
        throw new Error(JSON.stringify(res));
      }
      return res;
    })
    .catch((error) => {
      console.debug("retry failed", error);
      let retries_left = retries - 1;
      if (!retries_left) {
        throw error;
      }
      return fetch_with_retries(url, options, retries_left);
    });
}

function set_same_origin_headers(headers: Headers) {
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  return headers;
}

async function handle_request(event: FetchEvent) {
  const request = event.request
  const response = await fetch_with_retries(
    request.url,
    {
      method: request.method,
      headers: request.headers,
    },
    7
  );
  const headers = set_same_origin_headers(new Headers(response.headers));
  return new Response(response.body, {
    status: response.status,
    headers: headers,
  });
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