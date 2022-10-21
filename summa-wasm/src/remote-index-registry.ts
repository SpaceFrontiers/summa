import * as Comlink from "comlink";
import { IndexRegistry, IIndexRegistry, IndexRegistryOptions } from "./index-registry";
import { IndexAttributes, IndexEngineConfig } from "./grpc-web/index_service";
import {SearchRequest} from "./grpc-web/query";

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

    add(index_name: string, index_engine_config: IndexEngineConfig): Promise<IndexAttributes> {
        return this.index_registry.add(index_name, index_engine_config);
    }

    delete(index_name: string): Promise<void> {
        return this.index_registry.delete(index_name);
    }

    search(search_request: SearchRequest): Promise<object[]> {
        return this.index_registry.search(search_request)
    }

    search_by_binary_proto(search_request_proto_bytes: Uint8Array): Promise<object[]> {
        return this.index_registry.search_by_binary_proto(search_request_proto_bytes)
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
