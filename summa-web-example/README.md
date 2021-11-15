# An example web-interface using `summa-wasm`

### Install and launch example
npm i && npm run dev

### Important notes:

Interface is using `ServiceWorker` hackery for making search possible. You may want to use it too.
The issue is that IPFS does not emit following CORS headers:

- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Opener-Policy: same-origin`

and it may be worked around by setting headers inside `ServiceWorker`

These headers are required for enabling `SharedArrayBuffer` in browsers. `summa-wasm` needs `SharedArrayBuffer` because library spawns `WebWorkers` and
actively uses shared memory for passing data. 