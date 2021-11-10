type Header = {
    name: string;
    value: string;
};
export declare function request(method: string, url: string, headers: Array<{
    name: string;
    value: string;
}>): {
    data: Uint8Array;
    headers: Header[];
} | {
    status: number;
    status_text: string;
};
export declare function request_async(method: string, url: string, headers: Array<{
    name: string;
    value: string;
}>): Promise<unknown>;
export {};
