/// <reference types="vite/client" />

import { Summa } from "@/plugins/summa";
import { IPFSHTTPClient } from "ipfs-http-client";

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties  {
      summa: Summa
      ipfs: IPFSHTTPClient,
  }
}
