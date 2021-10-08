function wait_for_msg_type(target, type) {
  return new Promise(resolve => {
    target.addEventListener('message', function onMsg({ data }) {
      if (data == null || data.type !== type) return;
      target.removeEventListener('message', onMsg);
      resolve(data);
    });
  });
}

wait_for_msg_type(self, 'wasm_bindgen_worker_init').then(async data => {
  const pkg = await import('./');
  await pkg.default(data.module, data.memory);
  postMessage({ type: 'wasm_bindgen_worker_ready' });
  pkg.wbg_rayon_start_worker(data.receiver);
});

// Note: this is never used, but necessary to prevent a bug in Firefox
// (https://bugzilla.mozilla.org/show_bug.cgi?id=1702191) where it collects
// Web Workers that have a shared WebAssembly memory with the main thread,
// but are not explicitly rooted via a `Worker` instance.
//
// By storing them in a variable, we can keep `Worker` objects around and
// prevent them from getting GC-d.
let _workers;

export async function start_workers(module, memory, builder) {
  if (builder.num_threads() === 0) {
    throw new Error(`num_threads must be > 0.`);
  }

  const workerInit = {
    type: 'wasm_bindgen_worker_init',
    module,
    memory,
    receiver: builder.receiver()
  };
  _workers = await Promise.all(
    Array.from({ length: builder.num_threads() }, async () => {
      const worker = new Worker(self.location.href, { type: "module" });
      worker.postMessage(workerInit);
      await wait_for_msg_type(worker, 'wasm_bindgen_worker_ready');
      return worker;
    })
  );
  builder.build();
}