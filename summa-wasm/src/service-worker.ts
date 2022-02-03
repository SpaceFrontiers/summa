/// <reference lib="webworker" />
export default null
declare let self: ServiceWorkerGlobalScope

import Dexie from "dexie";

const CHUNK_SIZE = 16 * 1024;
interface ICacheItem {
    filename: string,
    chunk_id: number,
    blob: Uint8Array
}

class CacheDatabase extends Dexie {
    // Declare implicit table properties.
    // (just to inform Typescript. Instanciated by Dexie in stores() method)
    chunks!: Dexie.Table<ICacheItem, [string, number]>; // number = type of the primkey
    //...other tables goes here...

    constructor () {
        super("Cache");
        this.version(1).stores({
          chunks: "[filename+chunk_id]",
        });
    }
}

const db = new CacheDatabase();

function* generate_chunk_ids(start = 0, end = Infinity, step = 1) {
  let iterationCount = 0;
  for (let i = start; i < end; i += step) {
    iterationCount++;
    yield i;
  }
  return iterationCount;
}

async function set_from_cache(filename: string, start: number, end: number) {
  const chunk_ixs = Array.from(generate_chunk_ids(start, end, CHUNK_SIZE));
  try {
    const cached_chunks = await db.transaction("r!", db.chunks, async () => {
      return db.chunks
          .where("[filename+chunk_id]")
          .between(
              [filename, chunk_ixs[0]],
              [filename, chunk_ixs[chunk_ixs.length - 1]],
              true,
              true
          )
          .toArray();
    });
    if (cached_chunks.length < chunk_ixs.length) {
      return null;
    }
    const cached_response = new ArrayBuffer(end - start);
    let cached_response_view = new Uint8Array(cached_response);
    chunk_ixs.map((current, ix) => {
      cached_response_view.set(cached_chunks[ix].blob, current - start);
    });
    return cached_response_view;
  } catch (e) {
    console.error(e, filename, CHUNK_SIZE, start, end);
    throw e;
  }
}

async function fill_cache(response_body: ArrayBuffer, filename: string, start: number, end: number) {
  const items = Array.from(generate_chunk_ids(start, end, CHUNK_SIZE)).map(
    function (chunk_id) {
      const left_border = chunk_id - start;
      let right_border = left_border + CHUNK_SIZE;
      if (right_border >= end) {
        right_border = end;
      }
      return {
        filename: filename,
        chunk_id: chunk_id,
        blob: new Uint8Array(response_body.slice(left_border, right_border)),
      };
    }
  );
  db.transaction("rw!", db.chunks, async () => {
    await db.table("chunks").bulkPut(items);
  });
}

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
  let url = request.url;
  let filename = request.url;
  let [range_start, range_end] = [0, Infinity];
  const range = request.headers.get("range");
  if (range !== null) {
    // @ts-ignore
      const [_, start, end] = /^bytes=(\d+)-(\d+)?$/g.exec(range);
      range_start = parseInt(start);
      if (end) {
        range_end = parseInt(end) + 1;
      }
  }
  let response_body = null;
  let headers = new Headers();
  let status = 200;
  let is_immutable_file = filename.endsWith(".fast") ||
      filename.endsWith(".term") ||
      filename.endsWith(".pos") ||
      filename.endsWith(".store") ||
      filename.endsWith(".fieldnorm") ||
      filename.endsWith(".idx") ||
      filename.endsWith(".del");
  let caching_enabled = is_immutable_file &&
    request.method === "GET" &&
    range_end !== Infinity;
  if (caching_enabled) {
    response_body = await set_from_cache(filename, range_start, range_end);
    url += "?r=" + range_start;
  }
  if (response_body === null) {
    const response = await fetch_with_retries(
      url,
      {
        method: request.method,
        headers: request.headers,
      },
      7
    );
    response_body = await response.arrayBuffer();
    if (caching_enabled && response.ok) {
      fill_cache(
        response_body,
        filename,
        range_start,
        range_start + response_body.byteLength
      );
    }
    status = response.status;
    headers = new Headers(response.headers);
  }
  headers = set_same_origin_headers(headers);
  if (is_immutable_file) {
      headers.set("Cache-Control", "public, max-age=3600");
  }
  return new Response(response_body, {
    status: status,
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
