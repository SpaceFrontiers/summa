from typing import Dict, Optional

from izihawa_utils.pb_to_json import ParseDict

from .proto import search_service_pb2 as search_service_pb
from .summa_embed_bin import IndexRegistry


class SummaEmbed:
    def __init__(self):
        self.index_registry = IndexRegistry()

    def add(self, index_config, index_name: Optional[str] = None):
        parsed_index_config = search_service_pb.IndexQuery()
        ParseDict(index_config, parsed_index_config)
        return self.index_registry.add(parsed_index_config.SerializeToString(), index_name=index_name)

    def search(self, index_queries):
        search_request = search_service_pb.SearchRequest()
        for index_query in index_queries:
            if isinstance(index_query, Dict):
                dict_index_query = index_query
                index_query = search_service_pb.IndexQuery()
                ParseDict(dict_index_query, index_query)
            search_request.index_queries.append(index_query)
        return self.index_registry.search(search_request.SerializeToString())

