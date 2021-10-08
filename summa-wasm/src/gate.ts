export function request(method: string, url: string, headers: Array<{name: string, value: string}>): Uint8Array | {status: number, status_text: string} {
    var xhr = new XMLHttpRequest();
    xhr.responseType = "arraybuffer"
    xhr.open(method, url, false);
    if (headers !== undefined) {
        headers.forEach((header) => {
          xhr.setRequestHeader(header.name, header.value)
        });
    }
    xhr.send(null);
    if (xhr.status >= 200 && xhr.status < 300) {
        return new Uint8Array(xhr.response);
    } else {
        throw {
            status: xhr.status,
            status_text: xhr.statusText
        };
    }
}

export function request_async(method: string, url: string, headers: Array<{name: string, value: string}>) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.responseType = "arraybuffer"
        xhr.open(method, url);
         if (headers !== undefined) {
            headers.forEach((header) => {
              xhr.setRequestHeader(header.name, header.value)
            });
        }
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                let array = new Uint8Array(xhr.response);
                resolve(array);
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
    });
}

