---
title: IPFS Publish + WASM Browsing
parent: Guides
nav_order: 2
---
This guide covers three powerful technologies: [Iroh](https://github.com/n0-computer/iroh), [WASM](https://webassembly.org/getting-started/developers-guide/), and Summa. In this guide, you will learn how to:

Create a search index on the server
Replicate it to IPFS
Open and use it from inside your browser without the need for a server
Running a search engine in the browser means all computations required for the search will be done by your browser. The search engine will request data chunks over HTTP requests, and that's it. In the long run, this approach could significantly increase the privacy of search on the internet. If your search query doesn't leave your network, then nobody can track it.

Additionally, local processing of search queries would allow for full-featured search engines on statically hosted sites or even in decentralized systems such as IPFS.

### Configuring Summa and the index
First, you should set up Summa Server with Iroh Store (enabled by default) and create a test index using our [Quick-Start guide](/summa/guides/quick-start) 

### Publish the index to IPFS <a name="ipfs"></a>
To publish the index, we need to change its engine to IPFS. Then, Iroh P2P will automatically make it available to your IPFS peers:

```bash 
summa-cli localhost:8082 - migrate-index books books_iroh Ipfs
```
The command will return the CID of the published index that you can use later for replicating or opening it through the browser.
For example, you can find your index using `kubo`:

```bash
ipfs get <cid>
```

### Create web-application
It can be imported via URL into a browser:
```html
<script type="module">
  import * as Summa from "https://cdn.jsdelivr.net/npm/summa-wasm@0.98.0/dist/main.js";
</script>
```
Then, you will be able to instantiate your own search service:
```js
// IPFS hash of directory with the index of interest
// Replace it with your index!
const ipfs_hash = "bafybeigpui7vo3rstuyvicx5aeyve2n553lvczkiykj5nsl5e5rj6sb2gq";

// Directory URL that is used to access index
const directory_url = `http://localhost:8080/ipfs/${ipfs_hash}/`;

const worker_url = "https://cdn.jsdelivr.net/npm/summa-wasm@0.98.0/dist/root-worker.js";
const wasm_url = "https://cdn.jsdelivr.net/npm/summa-wasm@0.98.0/dist/index_bg.wasm"

// `remote_index_registry` is an object used to spawn threads for searching
// `setup` initializes WASM-module and pool of Web Workers.
const remote_index_registry = new Summa.RemoteIndexRegistry(worker_url, wasm_url, {num_threads: 4});
// Wait until workers will be set up
await remote_index_registry.init_guard;
```
Now, you have initialized search service that may be used for requesting remote indices:
```js
// `remote_engine_config` is a configuration object used for telling Summa how to reach remote index
const remote_engine_config = {
    method: "GET",
    url_template: `${directory_url}{file_name}`,
    headers_template: new Map([["range", "bytes={start}-{end}"]]),
    chunked_cache_config: { chunk_size: 16 * 1024, cache_size: 128 * 1024 * 1024 }
}

// Adding index to the worker makes is searchable.
await remote_index_registry.add(remote_engine_config, "test_index");
```
This is all! Just use it:
```js
const query = "Games of Thrones";
const index_query = {
    index_alias: "test_index",
    query: {query: {match: {value: query}}},
    collectors: [{collector: {top_docs: {limit: 5}}}],
}
const response = await remote_index_registry.search([ index_query ]);
console.log(response);
```

### Working example <a name="web"></a>
We have developed a working example that mirrors news agency feed to IPFS. Live example is available at `ipns://earthtimes.space`
Sources are committed to [GitHub](https://github.com/izihawa/earth-times) with a guide how to build it.
