from typing import (
    Dict,
    Optional,
)

from aiobaseclient import BaseStandardClient
from aiobaseclient.exceptions import ClientError
from izihawa_utils.common import filter_none

from .exceptions import (
    InvalidSyntaxError,
    QueryTimeoutError,
)


class SummaHttpClient(BaseStandardClient):
    def __init__(self, base_url: str, timeout: Optional[int] = None, ttl_dns_cache: Optional[int] = None):
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            ttl_dns_cache=ttl_dns_cache,
        )

    async def commit(self, schema: str, request_id: str = None):
        return await self.post(
            f'/v1/{schema}/commit/',
            headers={
                'Request-Id': request_id,
            },
        )

    async def put_document(self, schema: str, document: Dict, request_id: str = None):
        return await self.put(
            f'/v1/{schema}/',
            json=document,
            headers={
                'Content-Type': 'application/json',
                'Request-Id': request_id,
            },
        )

    async def search(
        self,
        schema: str,
        query: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        request_id: str = None,
    ):
        try:
            search_response = await self.get(
                f'/v1/{schema}/search/',
                params=filter_none({
                    'query': query,
                    'page': page,
                    'page_size': page_size,
                }),
                headers={
                    'Accept': 'application/json',
                    'Request-Id': request_id,
                },
            )
        except ClientError as e:
            if e.code == 'invalid_syntax_error':
                raise InvalidSyntaxError(query=query, page=page)
            elif e.code == 'timeout_error':
                raise QueryTimeoutError()
            raise e
        return search_response
