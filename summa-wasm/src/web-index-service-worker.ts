import * as Comlink from "comlink";
import { WebIndexService } from './web-index-service'

export const web_index_service = new WebIndexService();
Comlink.expose(web_index_service);
