import * as Comlink from "comlink";
import type {SummaW} from "@/workers/searcher";
import type { WebIndex } from '@/store/ipfs'
import {ipfs_url} from '@/plugins/ipfs'
import {sleep} from "@/utils";

const worker = new Worker(new URL("../workers/searcher.ts", import.meta.url), {
    type: "module",
});
const summa_worker = Comlink.wrap<SummaW>(worker);
await summa_worker.init();
export const summa = {
    ready_indices: new Set(),
    async setup(web_index: WebIndex) {
        await summa_worker.setup(
            JSON.parse(JSON.stringify(web_index)),
            "POST",
            `${ipfs_url}/api/v0/cat?arg=${web_index.path_hash}/{file_name}&offset={start}&length={length}`,
            null,
        );
        this.ready_indices.add(web_index.name);
    },
    async ready(name: string) {
        while (!this.ready_indices.has(name)) {
            await sleep(1000)
        }
    },
    async search(name: string, query: Object, collectors: Object[]) {
        await this.ready(name)
        return summa_worker.search(name, query, collectors)
    },
    stats: summa_worker.stats
}
export type Summa = typeof summa;