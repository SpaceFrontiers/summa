import { create } from "ipfs-http-client";

export const ipfs_url = "http://localhost:5001";
export const ipfs = create({ url: ipfs_url });