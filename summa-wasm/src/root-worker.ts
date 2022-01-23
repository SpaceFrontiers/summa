import * as Comlink from "comlink";
import { DefaultSearchService } from './default-search-service'

export const search_service = new DefaultSearchService();
Comlink.expose(search_service);