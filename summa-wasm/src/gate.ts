/* gats.ts contains routines that are used inside Rust code to make queries
* Both functions are required as there are use cases where only blocking
* synchronous calls are possible
*/

type Header = {name: string, value: string};

function parse_headers(xhr: XMLHttpRequest): Header[] {
    const headers = xhr.getAllResponseHeaders().toLowerCase();
    const arr = headers.trim().split(/[\r\n]+/);
    const headers_arr: Header[] = [];
    arr.forEach((line) => {
      const parts = line.split(': ');
      const name = parts.shift();
      const value = parts.join(': ');
      headers_arr.push({name: name!, value});
    });
    return headers_arr;
}

export function request(method: string, url: string, headers: {name: string, value: string}[], timeout_ms?: number): { data: Uint8Array, headers: Header[]} | {status: number, status_text: string} {
    var xhr = new XMLHttpRequest();
    xhr.responseType = "arraybuffer";
    xhr.withCredentials = true;
    if (timeout_ms) {
        xhr.timeout = timeout_ms;
    }
    try {
        xhr.open(method, url, false);
        if (headers !== undefined) {
            headers.forEach((header) => {
              xhr.setRequestHeader(header.name, header.value)
            });
        }
        xhr.send(null);
    } catch (e) {
        throw {
            status: 500,
            status_text: `${e}`,
        }
    }
    if (xhr.status >= 200 && xhr.status < 300) {
        return { data: new Uint8Array(xhr.response), headers: parse_headers(xhr) }
    } else {
        throw {
            status: xhr.status,
            status_text: xhr.statusText
        };
    }
}


export function request_async(method: string, url: string, headers: Array<{name: string, value: string}>, timeout_ms?: number) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.responseType = "arraybuffer";
        xhr.withCredentials = true;
        if (timeout_ms) {
            xhr.timeout = timeout_ms;
        }
        try {
            xhr.open(method, url);
            if (headers !== undefined) {
              headers.forEach((header) => {
                xhr.setRequestHeader(header.name, header.value)
              });
            }
            xhr.onload = function () {
                if (this.status >= 200 && this.status < 300) {
                    resolve({ data: new Uint8Array(xhr.response), headers: parse_headers(xhr) })
                } else {
                    reject({
                        status: this.status,
                        status_text: xhr.statusText
                    });
                }
            };
            xhr.onerror = function () {
                reject({
                    status: this.status,
                    status_text: xhr.statusText
                });
            };
            xhr.send();
        } catch (e) {
            reject({
                status: 500,
                status_text: `${e}`,
            })
        }
    });
}