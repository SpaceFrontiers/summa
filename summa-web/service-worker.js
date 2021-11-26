importScripts("dexie.min.js");

const CHUNK_SIZE = 16 * 1024;
const db = new Dexie("Cache");
db.version(5).stores({
  chunks: "[filename+chunk_id]",
});

function* generate_chunk_ids(start = 0, end = Infinity, step = 1) {
  let iterationCount = 0;
  for (let i = start; i <= end; i += step) {
    iterationCount++;
    yield i;
  }
  return iterationCount;
}

const fetch_with_retries = (url, options = {}, retries) =>
  fetch(url, options).then((res) => {
    if (!res.ok && retries > 0) {
      console.error("retry", url, "failed with", res);
      return fetch_with_retries(url, options, retries - 1);
    }
    return res;
  });

function set_same_origin_headers(headers) {
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  return headers;
}

async function set_from_cache(filename, chunk_size, start, end) {
  const cached_response = new ArrayBuffer(end - start + 1);
  const cached_response_view = new Uint8Array(cached_response);
  const chunk_ixs = Array.from(generate_chunk_ids(start, end, chunk_size));
  try {
    const cached_chunks = await db
      .table("chunks")
      .where("[filename+chunk_id]")
      .between(
        [filename, chunk_ixs[0]],
        [filename, chunk_ixs[chunk_ixs.length - 1]],
        true,
        true
      )
      .toArray();
    if (cached_chunks.length < chunk_ixs.length) {
      return null;
    }
    chunk_ixs.map((current, ix) => {
      cached_response_view.set(cached_chunks[ix].blob, current - start);
    });
    return cached_response_view;
  } catch (e) {
    console.error(e, filename, chunk_size, start, end);
    throw e;
  }
}

async function fill_cache(response_body, filename, chunk_size, start, end) {
  const items = await Promise.all(
    Array.from(generate_chunk_ids(start, end, chunk_size)).map(function (
      chunk_id
    ) {
      const left_border = chunk_id - start;
      let right_border = left_border + chunk_size;
      if (right_border >= end) {
        right_border = end + 1;
      }
      return {
        filename: filename,
        chunk_id: chunk_id,
        blob: new Uint8Array(response_body.slice(left_border, right_border)),
      };
    })
  );
  try {
    await db.table("chunks").bulkPut(items);
  } catch (e) {
    console.error(e, filename, chunk_size, start, end);
    throw e;
  }
}

function process_request_headers(request) {
  const new_headers = new Headers();
  let [range_start, range_end] = [null, null];
  let url = request.url;

  for (const [header, value] of request.headers) {
    if (header === "range") {
      const [_, start, end] = /^bytes=(\d+)-(\d+)$/g.exec(value);
      [range_start, range_end] = [parseInt(start), parseInt(end)];
    }
    new_headers.set(header, value);
  }
  const summa_cache_is_enabled =
    (url.endsWith(".json") ||
      url.endsWith(".term") ||
      url.endsWith(".store") ||
      url.endsWith(".idx") ||
      url.endsWith(".fast") ||
      url.endsWith(".pos") ||
      url.endsWith(".fieldnorm") ||
      url.endsWith(".bin")) &&
    request.method === "GET" &&
    range_start !== null &&
    range_end !== null;
  if (summa_cache_is_enabled) {
    url += "?r=" + range_end;
  }
  return {
    summa_cache_is_enabled: summa_cache_is_enabled,
    range_start: range_start,
    range_end: range_end,
    request: new Request(url, {
      method: request.method,
      headers: new_headers,
      cache: summa_cache_is_enabled ? "no-cache" : "default",
    }),
  };
}

async function handle_request(original_request) {
  const filename = original_request.url;
  let { summa_cache_is_enabled, range_start, range_end, request } =
    process_request_headers(original_request);
  console.log(summa_cache_is_enabled, range_start, range_end, filename)
  if (summa_cache_is_enabled) {
    const response_body = await set_from_cache(
      filename,
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
  let real_response = await fetch_with_retries(request, {}, 2);
  const real_response_body = await real_response.arrayBuffer();
  if (summa_cache_is_enabled && real_response.ok) {
    fill_cache(
      real_response_body,
      filename,
      CHUNK_SIZE,
      range_start,
      range_end
    ).catch((e) => console.error("Filling cache failed", e));
  }
  const new_headers = set_same_origin_headers(
    new Headers(real_response.headers)
  );
  new_headers.set("Cache-Control", "public, max-age=3600");
  return new Response(real_response_body, {
    status: real_response.status,
    statusText: real_response.statusText,
    headers: new_headers,
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
