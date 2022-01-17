export let _workers: Worker[] = [];

type WorkerData =  {
  type: string,
  module: WebAssembly.Module,
  memory: WebAssembly.Memory,
  state: any
}

function wait_for_msg_type(target: any, type: any): Promise<WorkerData> {
  return new Promise(resolve => {
    target.addEventListener('message', function onMsg(message: { data: WorkerData }) {
      if (message.data == null || message.data.type !== type) return;
      target.removeEventListener('message', onMsg);
      resolve(message.data);
    });
  });
}

export async function start_worker(module: WebAssembly.Module, memory: WebAssembly.Memory, state: any, opts: WorkerOptions) {
    const workerInit: WorkerData = {
        type: 'init',
        module,
        memory,
        state,
      };
    const worker = new Worker(self.location.href, opts);
    _workers.push(worker);
    worker.postMessage(workerInit);
    await wait_for_msg_type(worker, 'inited');
}

// Second: Entry script for the actual web worker.
if ('WorkerGlobalScope' in self &&
    self instanceof WorkerGlobalScope) {
    wait_for_msg_type(self, 'init').then(async (data: any) => {
      const pkg = await import('../pkg');
      await pkg.default(data.module, data.memory);
      postMessage({ type: 'inited' });
      pkg.worker_entry_point(data.state);
    });
}