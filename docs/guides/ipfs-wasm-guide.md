---
title: IPFS Publish + WASM Browsing
parent: Guides
nav_order: 2
---

### Configuring Summa and the index
First, you should set up Summa Server with Iroh Store (enabled by default) and create a test index using our [Quick-Start guide](/summa/quick-start) 

### Publish the index to IPFS
To publish the index, we need to change its engine to IPFS. Then, Iroh P2P will automatically make it available to your IPFS peers:

```bash 
summa-cli localhost:8082 - copy-index books books_iroh '{"ipfs": {}}'
```
The command will return the CID of the published index that you can use later for replicating or opening it through the browser.
For example, you can find your index using `kubo`:

```bash
ipfs get <cid>
```

### Create web-application
Firstly, you must set up an HTTP-server that may serve static content.

`summa-wasm` works in separate Web Worker to avoid blocking of the main thread and 
uses Service Worker for caching and mitigating some browser restrictions. Service Worker is an essential part
of the bundle and must be shipped together with your code. `summa-wasm` provides a service worker register script
that assumes you bundle `service-worker.js` file at the root of your static site.

```html
<head>
    ...
    <script type="module" src="./node_modules/summa-wasm/dist/service-worker-register.js"></script>
    <script type="module">
        import * as Summa from "https://cdn.jsdelivr.net/npm/summa-wasm@0.98.2/dist/main.js";
    </script>
</head>
```

Now we are ready to instantiate `summa-wasm`
```js
// URL of static HTTP-server (such as local nginx or CDN-hostings, or Storj, or IPFS, or whatever you want) 
// that supports `Range` queries and used to access index files
const directory_url = `http://localhost:8080/ipfs/${ipfs_hash}/`;

const worker_url = "https://cdn.jsdelivr.net/npm/summa-wasm@0.108.2/dist/root-worker.js";
const wasm_url = "https://cdn.jsdelivr.net/npm/summa-wasm@0.108.2/dist/index_bg.wasm"

// `remote_index_registry` is an object used to spawn threads for searching
// `setup` initializes WASM-module and pool of Web Workers.
const remote_index_registry = new Summa.RemoteIndexRegistry(worker_url, wasm_url);
// Wait until workers will be set up
await remote_index_registry.init_guard;
```

At this point, we have a working Summa inside your browser. Let's attach some network index to the `remote_index_registy`.

```js
// `remote_engine_config` is a configuration object used for telling Summa
// how to reach remote index
const remote_engine_config = {
    method: "GET",
    url_template: `https://our-fancy-domain.cdn-network.com/{file_name}`,
    headers_template: new Map([["range", "bytes={start}-{end}"]]),
}

// Adding index to the worker makes it searchable.
await remote_index_registry.add(remote_engine_config, "test_index");
```

`summa-wasm` will use this config for translating search engine's requests to files into network requests.
Then, when you will do a search:
```js
// All types of queries are supported
const index_query = {
    index_alias: "test_index",
    query: {query: {match: {value: "Game of Thrones"}}},
    collectors: [{collector: {top_docs: {limit: 5}}}],
}
const response = await remote_index_registry.search([ index_query ]);
```
`summa-wasm` will emit a pack of network requests for receiving only needed parts of the search index.

A search engine packed in WASM allows you to create semi-dynamic websites on static hosting or CDNs.
Given the well-developed infrastructure of CDN providers across the globe that can be used, it is possible
to bypass many barriers that may be built by censors.

### Working example

We have developed a working example that mirrors news agency feed to IPFS. Live example is available at `ipns://earthtimes.space`
Sources are committed to [GitHub](https://github.com/izihawa/earth-times) with a guide how to build it.
