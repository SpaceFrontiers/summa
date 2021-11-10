type WorkerData =  {
  type: string,
  module: WebAssembly.Module,
  memory: WebAssembly.Memory,
  receiver: any
}

function wait_for_msg_type(target: any, type: any): Promise<WorkerData> {
  return new Promise(resolve => {
    target.addEventListener('message', function onMsg(message: { data: WorkerData}) {
      if (message.data == null || message.data.type !== type) return;
      target.removeEventListener('message', onMsg);
      resolve(message.data);
    });
  });
}

wait_for_msg_type(self, 'wasm_bindgen_worker_init').then(async (data: WorkerData) => {
  const pkg = await import('../pkg');
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
export let _workers: Worker[];

export async function start_workers(module: WebAssembly.Module, memory: WebAssembly.Memory, builder: any) {
  if (builder.num_threads() === 0) {
    throw new Error(`num_threads must be > 0.`);
  }

  const workerInit: WorkerData = {
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