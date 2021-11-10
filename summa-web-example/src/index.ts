import * as Comlink from "comlink";

import { WebIndexServiceWorker } from "summa-wasm";

const ipfs_hash = "bafykbzacecb46ryjr3yf7yynmtxtuda7wyrwmvrzodlgaa5zuvy4iabhtbbsg";
const directory_url = `http://localhost:8080/ipfs/${ipfs_hash}/`;

const web_index_service_worker = Comlink.wrap<WebIndexServiceWorker>(
    new Worker(
        new URL(
          "summa-wasm/dist/worker.js",
          import.meta.url
        ),
        { type: "module" }
    )
);
const index_bg_wasm = new URL(
    "../node_modules/summa-wasm/dist/index_bg.wasm",
    import.meta.url
).href
await web_index_service_worker.setup(index_bg_wasm, 4)

const network_config = {
    method: "GET",
    url_template: `${directory_url}{file_name}`,
    headers_template: [{name: "range", value: "bytes={start}-{end}"}],
}

const index_payload = await web_index_service_worker.add(network_config);
const omnibox = document.getElementById("omnibox");

omnibox.onkeydown = ((e) => {
    if(e.code == "Enter") {
        const index_query = {
            index_name: index_payload["name"],
            query: {query: {match: {value: (omnibox as HTMLInputElement).value}}},
            collectors: [{collector: {top_docs: {limit: 5, offset: 0, fields: [], snippets: [], explain: false}}}],
        }
        web_index_service_worker.search([ index_query ]).then((search_results) => {
            document.getElementById("search_result").innerText = JSON.stringify(
                search_results[0]["collector_output"]["top_docs"]["scored_documents"].map((el) => JSON.parse(el.document)["title"])
            );
        })
    }
});

