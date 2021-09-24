import { defineStore } from "pinia";
import type { PeerId } from "ipfs-http-client/dht/map-event";
import type { IPFSPath } from "ipfs-core-types/dist/src/utils";
import { ipfs } from "@/plugins/ipfs"

export class WebIndex {
  name: string;
  name_hash: PeerId;
  path_hash: IPFSPath | undefined;
  default_fields: string[];
  multi_fields: string[];
  files: {Name: string, Size: number}[];
  size: number;
  local_size: number;

  constructor(name: string, name_hash: PeerId, default_fields: string[], multi_fields: string[]) {
    this.name = name
    this.name_hash = name_hash
    this.path_hash = undefined
    this.default_fields = default_fields
    this.multi_fields = multi_fields
    this.files = []
    this.size = 0;
    this.local_size = 0;
  }
}

// @ts-ignore
export const useIpfsStore = defineStore("ipfs", {
  state: () => ({ web_indices: [] as WebIndex[]}),
  actions: {
    async setup(name: string, name_hash: PeerId, default_fields: string[], multi_fields: string[]): Promise<WebIndex> {
      // @ts-ignore
      let web_index = this.lookup(name);
      if (web_index === undefined) {
        web_index = new WebIndex(name, name_hash, default_fields, multi_fields)
        this.web_indices.push(web_index)
        // @ts-ignore
        await this.update(name)
      }
      return web_index;
    },
    lookup(name: string): WebIndex {
      let filtered_web_indices = this.web_indices.filter((web_index) => web_index.name == name);
      return filtered_web_indices[0] as WebIndex
    },
    async update_size(name: string) {
      // @ts-ignore
      let web_index = this.lookup(name);
      let stat = await ipfs.files.stat(web_index.path_hash, {
        withLocal: true,
        size: true
      });
      web_index.size = stat.cumulativeSize;
      web_index.local_size = stat.sizeLocal;
    },
    async update(name: string) {
      // @ts-ignore
      let web_index = this.lookup(name);
      let last_response = undefined;
      for await (const response of ipfs.name.resolve(web_index.name_hash)) {
        last_response = response;
      }
      web_index.path_hash = last_response;
      web_index.files = [];
      for await (const file of ipfs.ls(web_index.path_hash)) {
        web_index.files.push({Name: file.name, Size: file.size});
      }
      // @ts-ignore
      await this.update_size(name);
    }
  },
  persist: true,
});
