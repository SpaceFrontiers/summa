import re
from typing import Dict
from urllib.parse import urlparse

from aiokit import AioThing
from izihawa_utils.pb_to_json import ParseDict

from .proto import index_service_pb2, query_pb2, search_service_pb2
from .summa_embed_bin import IndexRegistry as IndexRegistryBin


async def detect_host_header(url, aiohttp=None):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, allow_redirects=False) as resp:
            if 300 <= resp.status < 400:
                redirection_url = resp.headers['Location']
                if 'localhost' in redirection_url:
                    parsed_url = urlparse(redirection_url)
                    return re.search(r'(.*)\.localhost.*', parsed_url.netloc).group(0)


def canonoize_endpoint(endpoint):
    endpoint = endpoint.rstrip('/')
    if not endpoint.startswith('http'):
        endpoint = 'http://' + endpoint
    return endpoint


class SummaEmbedClient(AioThing):
    def __init__(self):
        super().__init__()
        self.index_registry = IndexRegistryBin()

    async def add_remote_index(self, index_name, full_path, cache_size, query_parser_config=None):
        headers_template = {'range': 'bytes={start}-{end}'}
        if host_header := await detect_host_header(full_path):
            headers_template['host'] = host_header
        index_engine_config = {
            'remote': {
                'method': 'GET',
                'url_template': f'{full_path}{{file_name}}',
                'headers_template': headers_template,
                'cache_config': {'cache_size': cache_size},
            },
            'query_parser_config': query_parser_config
        }
        return await self.add(index_engine_config, index_name=index_name)

    async def add_local_index(self, index_name, full_path, query_parser_config=None):
        index_engine_config = {
            'file': {
                'path': full_path,
            },
            'query_parser_config': query_parser_config
        }
        return await self.add(index_engine_config, index_name=index_name)

    async def add(self, index_engine_config, index_name: str) -> index_service_pb2.IndexAttributes:
        parsed_index_engine_config = index_service_pb2.IndexEngineConfig()
        ParseDict(index_engine_config, parsed_index_engine_config)
        index_attributes_bytes = await self.index_registry.add(
            parsed_index_engine_config.SerializeToString(),
            index_name=index_name,
        )
        index_attributes = index_service_pb2.IndexAttributes()
        index_attributes.ParseFromString(index_attributes_bytes)
        return index_attributes

    async def search(self, index_queries) -> query_pb2.SearchResponse:
        search_request = search_service_pb2.SearchRequest()
        for index_query in index_queries:
            if isinstance(index_query, Dict):
                dict_index_query = index_query
                index_query = search_service_pb2.IndexQuery()
                ParseDict(dict_index_query, index_query)
            search_request.index_queries.append(index_query)
        search_response_bytes = await self.index_registry.search(search_request.SerializeToString())
        search_response = query_pb2.SearchResponse()
        search_response.ParseFromString(search_response_bytes)
        return search_response
