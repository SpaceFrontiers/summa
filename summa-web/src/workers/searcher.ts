import * as Comlink from "comlink";
import type { WebIndex } from '@/store/ipfs'
import init, { SummaIndexWrapper } from "summa-wasm";
import wasm_url from "summa-wasm/index_bg.wasm?url";
import { stats } from "summa-wasm/gate";

const index_cache = new Map<string, SummaIndexWrapper>();


export const summa_w = {
  pending_setup: new Set(),
  async init() {
      await init(wasm_url);
  },
  async setup(
    web_index: WebIndex,
    method: string,
    url_template: string,
    headers_template: Map<string, string> | null,
  ) {
    index_cache.set(
      web_index.name,
      new SummaIndexWrapper(
          method,
          url_template,
          headers_template,
          web_index.files,
          web_index.default_fields.join(','),
          web_index.multi_fields.join(',')
      )
    );
  },
  search(name: string, query: Object, collectors: Object[]) {
    return index_cache.get(name)!.search(name, query, collectors);
  },
  stats() {
    return stats();
  },
};
export type SummaW = typeof summa_w;

Comlink.expose(summa_w);
self.postMessage("ready");
