importScripts("localforage.js");

function* range(start = 0, end = Infinity, step = 1) {
  let iterationCount = 0;
  for (let i = start; i < end; i += step) {
    iterationCount++;
    yield i < end ? i : end;
  }
  return iterationCount;
}

function set_same_origin_headers(headers) {
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  return headers;
}

const CHUNK_SIZE = 16384;

async function fill_from_cache(request, range_header) {
  const [start, end] = range_header;
  const cached_response = new ArrayBuffer(end - start + 1);
  const cached_response_view = new Uint8Array(cached_response);
  for (const current of range(start, end, CHUNK_SIZE)) {
    const url = new URL(request.url).pathname;
    const cache_key = `cache:${url}:${current}-${current + CHUNK_SIZE}`;
    const cached = await localforage.getItem(cache_key);
    if (cached === null) {
      return null;
    }
    cached_response_view.set(cached, current - start);
  }
  return cached_response;
}

async function fill_cache(request, response_body, range_header) {
  const [start, end] = range_header;
  for (const current of range(start, end, CHUNK_SIZE)) {
    const url = new URL(request.url).pathname;
    const cache_key = `cache:${url}:${current}-${current + CHUNK_SIZE}`;
    await localforage.setItem(
      cache_key,
      new Uint8Array(
        response_body.slice(current - start, current - start + CHUNK_SIZE)
      )
    );
  }
}

function process_request_headers(request) {
  const new_headers = new Headers();
  let range_header = null;
  let is_summa_cache = false;
  for (const [header, value] of request.headers) {
    if (header === "x-summa-cache") {
      is_summa_cache = true;
      continue;
    }
    if (header === "range") {
      const [_, start, end] = /^bytes=(\d+)-(\d+)$/g.exec(value);
      range_header = [parseInt(start), parseInt(end)];
    }
    new_headers.set(header, value);
  }
  return {
    is_summa_cache: is_summa_cache,
    range_header: range_header,
    request: new Request(request.url, {
      method: request.method,
      headers: new_headers,
      cache: "no-store",
    }),
  };
}

async function handle_request(original_request) {
  const { is_summa_cache, range_header, request } =
    process_request_headers(original_request);
  if (is_summa_cache && range_header) {
    const response_body = await fill_from_cache(request, range_header);
    if (response_body) {
      return new Response(response_body, {
        headers: set_same_origin_headers(new Headers()),
      });
    }
  }
  const real_response = await fetch(request);
  const real_response_body = await real_response.arrayBuffer();
  if (is_summa_cache && range_header) {
    fill_cache(request, real_response_body, range_header).catch((e) =>
      console.error("Filling cache failed", e)
    );
  }
  return new Response(real_response_body, {
    status: real_response.status,
    statusText: real_response.statusText,
    headers: set_same_origin_headers(new Headers(real_response.headers)),
  });
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) =>
  event.waitUntil(self.clients.claim())
);

self.addEventListener("message", (ev) => {
  if (ev.data && ev.data.type === "deregister") {
    self.registration
      .unregister()
      .then(() => {
        return self.clients.matchAll();
      })
      .then((clients) => {
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
  event.respondWith(handle_request(event.request));
});