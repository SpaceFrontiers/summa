import * as Comlink from "comlink";

import { DefaultSearchService } from "summa-wasm";

// IPFS hash of directory with the index of interest
// Replace it with your index!
const ipfs_hash = "bafybeigpui7vo3rstuyvicx5aeyve2n553lvczkiykj5nsl5e5rj6sb2gq";

// Directory URL that is used to access index
const directory_url = `http://localhost:8080/ipfs/${ipfs_hash}/`;

// Queries are done through a separate Web Worker
const web_index_service_worker = Comlink.wrap<DefaultSearchService>(
    new Worker(
        new URL(
          "summa-wasm/dist/root-worker.js",
          import.meta.url
        ),
        { type: "module" }
    )
);
const index_bg_wasm = new URL(
    "summa-wasm/dist/index_bg.wasm",
    import.meta.url
).href
// `web_index_service_worker` is an object used to spawn threads for searching
// `setup` initializes WASM-module and pool of Web Workers.
await web_index_service_worker.setup(index_bg_wasm, 4)

// `remote_engine_config` is a configuration object used for telling Summa how to reach remote index
const remote_engine_config = {
    method: "GET",
    url_template: `${directory_url}{file_name}`,
    headers_template: new Map([["range", "bytes={start}-{end}"]]),
    chunked_cache_config: { chunk_size: 16 * 1024, cache_size: 128 * 1024 * 1024 }
}
await web_index_service_worker.add(remote_engine_config, "test_index");
const omnibox = document.getElementById("omnibox");

omnibox.onkeydown = ((e) => {
    if(e.code == "Enter") {
        const query = (omnibox as HTMLInputElement).value;
        console.log("Sending query", query);
        const index_query = {
            index_alias: "test_index",
            query: {query: {match: {value: query}}},
            collectors: [{collector: {top_docs: {limit: 5}}}],
        }
        web_index_service_worker.search([ index_query ]).then((search_results) => {
            document.getElementById("search_result").innerText = JSON.stringify(
                search_results[0]["collector_output"]["top_docs"]["scored_documents"].map((el) => JSON.parse(el.document)["title"])
            );
        })
    }
});

