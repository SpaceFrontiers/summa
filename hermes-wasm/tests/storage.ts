export type IFilesStorage = {
	write: (uuid: string, buffer: ArrayBuffer) => Promise<void>;
	get: (uuid: string) => Promise<ArrayBuffer | null>;
	delete: (uuid: string[]) => Promise<void>;
	list: () => Promise<string[]>;
};

export class InMemoryFS implements IFilesStorage {
	protected readonly storage: Record<string, ArrayBuffer> = {};
	constructor(initData: Record<string, ArrayBuffer> = {}) {
		this.storage = {};

		for (const [id, buffer] of Object.entries(initData)) {
			this.storage[id] = new Uint8Array(buffer).buffer;
		}
	}

	write = async (id: string, buffer: ArrayBuffer) => {
		this.storage[id] = new Uint8Array(buffer).buffer;
	};

	get = async (id: string) => {
		return this.storage[id];
	};

	delete = async (ids: string[]) => {
		ids.forEach((id) => {
			delete this.storage[id];
		});
	};

	list = async () => {
		return Object.keys(this.storage);
	};
}
