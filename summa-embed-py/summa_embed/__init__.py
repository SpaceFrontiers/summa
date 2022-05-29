from typing import Dict, Optional

from izihawa_utils.pb_to_json import ParseDict

from .proto import index_service_pb2, search_service_pb2
from .summa_embed_bin import IndexRegistry as IndexRegistryBin


class IndexRegistry:
    def __init__(self):
        self.index_registry = IndexRegistryBin()

    def add(self, index_config, index_name: Optional[str] = None):
        parsed_index_config = index_service_pb2.IndexEngineConfig()
        ParseDict(index_config, parsed_index_config)
        return self.index_registry.add(parsed_index_config.SerializeToString(), index_name=index_name)

    def search(self, index_queries):
        search_request = search_service_pb2.SearchRequest()
        for index_query in index_queries:
            if isinstance(index_query, Dict):
                dict_index_query = index_query
                index_query = search_service_pb2.IndexQuery()
                ParseDict(dict_index_query, index_query)
            search_request.index_queries.append(index_query)
        return self.index_registry.search(search_request.SerializeToString())
