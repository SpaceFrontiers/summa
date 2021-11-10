export declare class NetworkConfig {
    method: string;
    url_template: string;
    headers_template: {
        name: string;
        value: string;
    }[] | null;
    chunked_cache_config?: ChunkedCacheConfig;
    constructor(method: string, url_template: string, headers_template: {
        name: string;
        value: string;
    }[] | null, chunked_cache_config?: ChunkedCacheConfig);
}
export declare class ChunkedCacheConfig {
    chunk_size: number;
    cache_size?: number;
    constructor(chunk_size: number, cache_size?: number);
}
