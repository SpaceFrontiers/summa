importScripts("localforage.js");

const CHUNK_SIZE = 16384;

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

function get_chunk_cache_key(cache_key, chunk_id, chunk_size) {
  return `${cache_key}:${chunk_id}-${chunk_id + chunk_size}`;
}

async function fill_from_cache(cache_key, chunk_size, start, end) {
  const cached_response = new ArrayBuffer(end - start + 1);
  const cached_response_view = new Uint8Array(cached_response);
  for (const current of range(start, end, chunk_size)) {
    const chunk_cache_key = get_chunk_cache_key(cache_key, current, chunk_size);
    const cached = await localforage.getItem(chunk_cache_key);
    if (cached === null) {
      return null;
    }
    cached_response_view.set(cached, current - start);
  }
  return cached_response;
}

async function fill_cache(response_body, cache_key, chunk_size, start, end) {
  for (const current of range(start, end, chunk_size)) {
    const chunk_cache_key = get_chunk_cache_key(cache_key, current, chunk_size);
    await localforage.setItem(
      chunk_cache_key,
      new Uint8Array(
        response_body.slice(current - start, current - start + chunk_size)
      )
    );
  }
}

function process_request_headers(request) {
  const new_headers = new Headers();
  let [range_start, range_end] = [null, null];
  let summa_cache_is_enabled = false;
  for (const [header, value] of request.headers) {
    if (header === "x-summa-cache-is-enabled") {
      summa_cache_is_enabled = true;
      continue;
    }
    if (header === "range") {
      const [_, start, end] = /^bytes=(\d+)-(\d+)$/g.exec(value);
      [range_start, range_end] = [parseInt(start), parseInt(end)];
    }
    new_headers.set(header, value);
  }
  return {
    summa_cache_is_enabled: summa_cache_is_enabled,
    range_start: range_start,
    range_end: range_end,
    request: new Request(request.url, {
      method: request.method,
      headers: new_headers,
      cache: "no-store",
    }),
  };
}

async function handle_request(original_request) {
  const {
    summa_cache_is_enabled,
    range_start,
    range_end,
    request,
  } = process_request_headers(original_request);
  const cache_key = `cache:${new URL(request.url).pathname}`;
  if (summa_cache_is_enabled) {
    const response_body = await fill_from_cache(
      cache_key,
      CHUNK_SIZE,
      range_start,
      range_end
    );
    if (response_body) {
      return new Response(response_body, {
        headers: set_same_origin_headers(new Headers()),
      });
    }
  }
  const real_response = await fetch(request);
  const real_response_body = await real_response.arrayBuffer();
  if (summa_cache_is_enabled) {
    fill_cache(
      real_response_body,
      cache_key,
      CHUNK_SIZE,
      range_start,
      range_end
    ).catch((e) => console.error("Filling cache failed", e));
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
