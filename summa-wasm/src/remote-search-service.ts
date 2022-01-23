import * as Comlink from "comlink";
import {IndexQuery, SearchService} from "./search-service";
import {DefaultSearchService} from "./default-search-service";
import {RemoteEngineConfig, IndexAttributes} from "./configs";

export class RemoteSearchService implements SearchService {
    init_guard: {
        promise: Promise<void>;
    };
    web_index_service_worker: Comlink.Remote<DefaultSearchService>;

    constructor(worker_url: URL, wasm_url: URL, options: { num_threads: number }) {
        this.web_index_service_worker = Comlink.wrap<DefaultSearchService>(
            new Worker(
                worker_url,
                {type: "module"}
            )
        );
        this.init_guard = {
            promise: this.setup(wasm_url, options),
        };
    }

    add(remote_engine_config: RemoteEngineConfig, index_name?: string | undefined): Promise<IndexAttributes> {
        return this.web_index_service_worker.add(remote_engine_config, index_name);
    }

    delete(index_name: string): Promise<void> {
        return this.web_index_service_worker.delete(index_name);
    }

    search(index_queries: IndexQuery[]): Promise<object[]> {
        return this.web_index_service_worker.search(index_queries)
    }

    warmup(index_name: string): Promise<void> {
        return this.web_index_service_worker.warmup(index_name);
    }

    index_document(index_name: string, document: string): Promise<void> {
        return this.web_index_service_worker.index_document(index_name, document)
    }

    commit(index_name: string): Promise<void> {
        return this.web_index_service_worker.commit(index_name);
    }

    async setup(wasm_url: URL, options: { num_threads: number }) {
        return await this.web_index_service_worker.setup(
            wasm_url.href,
            options.num_threads
        );
    }
}
