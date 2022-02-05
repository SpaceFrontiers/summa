import * as Comlink from "comlink";
import {IndexRegistry, IIndexRegistry, IndexQuery} from "./index-registry";
import {IndexAttributes, IndexEngineConfig} from "./configs";

export class RemoteIndexRegistry implements IIndexRegistry {
    init_guard: Promise<void>;
    search_service: Comlink.Remote<IndexRegistry>;

    constructor(worker_url: URL, wasm_url: URL, options: { num_threads: number }) {
        this.search_service = Comlink.wrap<IndexRegistry>(
            new Worker(
                worker_url,
                { type: "module" }
            )
        );
        this.init_guard = this.setup(wasm_url, options);
    }

    add(index_engine_config: IndexEngineConfig, index_name?: string | undefined): Promise<IndexAttributes> {
        return this.search_service.add(index_engine_config, index_name);
    }

    delete(index_name: string): Promise<void> {
        return this.search_service.delete(index_name);
    }

    search(index_queries: IndexQuery[]): Promise<object[]> {
        return this.search_service.search(index_queries)
    }

    warmup(index_name: string): Promise<void> {
        return this.search_service.warmup(index_name);
    }

    index_document(index_name: string, document: string): Promise<void> {
        return this.search_service.index_document(index_name, document)
    }

    commit(index_name: string): Promise<void> {
        return this.search_service.commit(index_name);
    }

    async setup(wasm_url: URL, options: { num_threads: number }) {
        return await this.search_service.setup(
            wasm_url.href,
            options.num_threads
        );
    }
}
