import { WebIndexRegistry } from "../pkg";
import { NetworkConfig } from "./configs";
export type StatusCallback = (type: string, message: string) => void;
export declare class IndexQuery {
    index_name: string;
    query: Object;
    collectors: Object[];
    constructor(index_name: string, query: Object, collectors: Object[]);
}
export declare class WebIndexServiceWorker {
    registry?: WebIndexRegistry;
    setup(init_url: string, threads: number, status_callback?: StatusCallback): Promise<void>;
    add(network_config: NetworkConfig): Promise<Object>;
    delete(index_name: string): Promise<void>;
    search(index_queries: IndexQuery[]): Promise<any>;
    cache_metrics(): Promise<any>;
    warmup(index_name: string): Promise<void>;
}
export declare const web_index_service_worker: WebIndexServiceWorker;
