const statistics = {
    downloaded_bytes: 0,
    requests: 0
}

export function stats() {
    return statistics;
}

export function request(method: string, url: string, headers: Map<string, string>) {
    var xhr = new XMLHttpRequest();
    xhr.responseType = "arraybuffer"
    if (headers !== undefined) {
        headers.forEach((header_name, header_value) => {
          xhr.setRequestHeader(header_name, header_value)
        });
    }
    xhr.open(method, url, false);
    xhr.send(null);
    let array = new Uint8Array(xhr.response);
    statistics.downloaded_bytes += array.byteLength;
    statistics.requests += 1;
    return array;
}

