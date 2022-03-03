import * as Comlink from "comlink";
import {IndexRegistry, IIndexRegistry, IndexQuery, IndexRegistryOptions} from "./index-registry";
import {IndexAttributes, IndexEngineConfig} from "./configs";

export class RemoteIndexRegistry implements IIndexRegistry {
    init_guard: Promise<void>;
    index_registry: Comlink.Remote<IndexRegistry>;

    constructor(worker_url: URL, wasm_url: URL, options: IndexRegistryOptions) {
        this.index_registry = Comlink.wrap<IndexRegistry>(
            new Worker(
                worker_url,
                { type: "module" }
            )
        );
        this.init_guard = this.setup(wasm_url, options);
    }

    add(index_engine_config: IndexEngineConfig, index_name?: string | undefined): Promise<IndexAttributes> {
        return this.index_registry.add(index_engine_config, index_name);
    }

    delete(index_name: string): Promise<void> {
        return this.index_registry.delete(index_name);
    }

    search(index_queries: IndexQuery[]): Promise<object[]> {
        return this.index_registry.search(index_queries)
    }

    warmup(index_name: string): Promise<void> {
        return this.index_registry.warmup(index_name);
    }

    index_document(index_name: string, document: string): Promise<void> {
        return this.index_registry.index_document(index_name, document)
    }

    commit(index_name: string): Promise<void> {
        return this.index_registry.commit(index_name);
    }

    extract_terms(index_name: string, field_name: string, limit: number, start_from?: string): Promise<string[]> {
        return this.index_registry.extract_terms(index_name, field_name, limit, start_from);
    }

    get_index_field_names(index_name: string): Promise<string[]> {
        return this.index_registry.get_index_field_names(index_name);
    }

    async setup(
        wasm_url: URL,
        options: IndexRegistryOptions
    ) {
        return await this.index_registry.setup(
            wasm_url.href,
            options,
        );
    }
}
