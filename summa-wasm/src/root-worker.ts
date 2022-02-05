import * as Comlink from "comlink";
import { IndexRegistry } from './index-registry'

export const search_service = new IndexRegistry();
Comlink.expose(search_service);