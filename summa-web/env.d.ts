/// <reference types="vite/client" />

import { IPFSHTTPClient } from "ipfs-http-client";
import { WebIndexService } from "@/services/summa";

declare module "@vue/runtime-core" {
  interface ComponentCustomProperties {
    ipfs: IPFSHTTPClient;
    web_index_service: WebIndexService;
  }
}

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      GITHUB_AUTH_TOKEN: string;
      NODE_ENV: "development" | "production";
      PORT?: string;
      PWD: string;
    }
  }
}