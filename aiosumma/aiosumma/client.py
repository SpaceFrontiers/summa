import asyncio
import sys
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple, Union

import grpc
import orjson as json
from aiogrpcclient import BaseGrpcClient, expose
from grpc import StatusCode
from grpc.experimental.aio import AioRpcError
from izihawa_utils.pb_to_json import ParseDict

from .proto import consumer_service_pb2 as consumer_service_pb
from .proto import index_service_pb2 as index_service_pb
from .proto import query_pb2 as query_pb
from .proto import reflection_service_pb2 as reflection_service_pb
from .proto import search_service_pb2 as search_service_pb
from .proto.consumer_service_pb2_grpc import ConsumerApiStub
from .proto.index_service_pb2_grpc import IndexApiStub
from .proto.reflection_service_pb2_grpc import ReflectionApiStub
from .proto.search_service_pb2_grpc import SearchApiStub
from .proto.utils_pb2 import Asc, Desc  # noqa


def setup_metadata(session_id, request_id):
    metadata = []
    if session_id:
        metadata.append(('session-id', session_id))
    if request_id:
        metadata.append(('request-id', request_id))
    return metadata


def documents_portion_iter(index_name: str, documents: Iterable, bulk_size: int, conflict_strategy: Optional[str] = None):
    documents_portion = []
    for document in documents:
        documents_portion.append(document)
        if len(documents_portion) > bulk_size:
            yield index_service_pb.IndexDocumentStreamRequest(
                index_name=index_name,
                documents=documents_portion,
                conflict_strategy=conflict_strategy,
            )
            documents_portion = []
    if documents_portion:
        yield index_service_pb.IndexDocumentStreamRequest(
            index_name=index_name,
            documents=documents_portion,
            conflict_strategy=conflict_strategy,
        )

class SummaClient(BaseGrpcClient):
    stub_clses = {
        'consumer_api': ConsumerApiStub,
        'index_api': IndexApiStub,
        'reflection_api': ReflectionApiStub,
        'search_api': SearchApiStub,
    }

    @expose
    async def attach_index(
            self,
            index_name: str,
            index_engine: dict,
            merge_policy: Optional[Dict] = None,
            query_parser_config: Optional[Dict] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.AttachIndexResponse:
        """
        Attach index to Summa. It must be placed under data directory named as `index_name`

        Args:
            index_name: index name
            index_engine: {"file": {}}, {"memory": {}} or {"remote": {"cache_config": {"cache_size": ...}}}
            merge_policy: describes how to select segments for merging, possible values: {"log": {}}, {"temporal": {}}
            query_parser_config: describes how to parse queries by default.
            request_id: request id
            session_id: session id
        """
        query_parser_config_pb = query_pb.QueryParserConfig()
        if query_parser_config:
            ParseDict(query_parser_config, query_parser_config_pb)
        return await self.stubs['index_api'].attach_index(
            index_service_pb.AttachIndexRequest(
                index_name=index_name,
                merge_policy=index_service_pb.MergePolicy(**(merge_policy or {})),
                query_parser_config=query_parser_config_pb,
                **index_engine,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def commit_index(
            self,
            index_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.CommitIndexResponse:
        """
        Commit index asynchronously. A commit will be scheduled and be done eventually.
        Executing commit means stopping all consumption before commit and starting it again
        after.

        Args:
            index_name: index name
            request_id: request id
            session_id: session id
        Returns:
            Commit scheduling result
        """
        return await self.stubs['index_api'].commit_index(
            index_service_pb.CommitIndexRequest(
                index_name=index_name,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def copy_documents(
            self,
            source_index_name: str,
            target_index_name: str,
            conflict_strategy: Optional[str] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.CopyDocumentsResponse:
        """
        Copies all documents from `source` to `target` index

        Args:
            source_index_name: source index name
            target_index_name: target index name
            conflict_strategy: recommended to set to DoNothing for large updates and maintain uniqueness in your application
            request_id: request id
            session_id: session id
        """
        if isinstance(conflict_strategy, str):
            conflict_strategy = index_service_pb.ConflictStrategy.Value(conflict_strategy)
        return await self.stubs['index_api'].copy_documents(
            index_service_pb.CopyDocumentsRequest(
                source_index_name=source_index_name,
                target_index_name=target_index_name,
                conflict_strategy=conflict_strategy,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def create_consumer(
            self,
            index_name: str,
            consumer_name: str,
            bootstrap_servers: List[str],
            group_id: str,
            topics: List[str],
            request_id: str = None,
            session_id: str = None,
    ) -> consumer_service_pb.CreateConsumerResponse:
        """
        Create consumer and corresponding topics in Kafka.
        The newly created consumer starts immediately after creation

        Args:
            index_name: index name
            consumer_name: consumer name that will be used for topic creation in Kafka too
            bootstrap_servers: list of bootstrap servers
            group_id: group_id for Kafka topic consumption
            topics: list of topics
            request_id: request id
            session_id: session id
        """
        return await self.stubs['consumer_api'].create_consumer(
            consumer_service_pb.CreateConsumerRequest(
                consumer_name=consumer_name,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                index_name=index_name,
                topics=topics,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose(with_from_file=True)
    async def create_index(
            self,
            index_name: str,
            schema: Union[str, list],
            index_engine: dict,
            compression: Optional[Union[str, int]] = None,
            blocksize: Optional[int] = None,
            sort_by_field: Optional[Tuple] = None,
            index_attributes: Optional[Dict] = None,
            merge_policy: Optional[Dict] = None,
            query_parser_config: Optional[Dict] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.CreateIndexResponse:
        """
        Create index

        Args:
            index_name: index name
            index_engine: {"file": {}}, {"memory": {}} or {"ipfs": {"cache_config": {"cache_size": ...}}}
            schema: Tantivy index schema
            compression: Tantivy index compression
            blocksize: Docstore blocksize
            index_attributes: Various index attributes such as default fields, multi fields, primary keys etc.
            merge_policy: describes how to select segments for merging, possible values: {"log": {}}, {"temporal": {}}
            query_parser_config: describes how to parse queries by default.
            request_id: request id
            session_id: session id
            sort_by_field: (field_name, order)
        """
        if isinstance(compression, str):
            compression = index_service_pb.Compression.Value(compression)
        elif isinstance(compression, int):
            compression = index_service_pb.Compression.Name(compression)
        if isinstance(schema, list):
            schema = json.dumps(schema)

        query_parser_config_pb = query_pb.QueryParserConfig()
        if query_parser_config:
            ParseDict(query_parser_config, query_parser_config_pb)

        return await self.stubs['index_api'].create_index(
            index_service_pb.CreateIndexRequest(
                index_name=index_name,
                schema=schema,
                compression=compression,
                blocksize=blocksize,
                index_attributes=index_attributes,
                sort_by_field=index_service_pb.SortByField(
                    field=sort_by_field[0],
                    order=sort_by_field[1].capitalize(),
                ) if sort_by_field else None,
                merge_policy=index_service_pb.MergePolicy(**(merge_policy or {})),
                query_parser_config=query_parser_config_pb,
                **index_engine
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose(with_from_file=True)
    async def copy_index(
            self,
            source_index_name: str,
            target_index_name: str,
            target_index_engine: dict,
            merge_policy: Optional[Dict] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.CreateIndexResponse:
        """
        Copies existing index to new engine

        Args:
            source_index_name: source index name
            target_index_name: target index name
            target_index_engine: {"file": {}}, {"memory": {}} or {"ipfs": {"cache_config": {"cache_size": ...}}}
            merge_policy: describes how to select segments for merging, possible values: {"log": {}}, {"temporal": {}}
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].copy_index(
            index_service_pb.CopyIndexRequest(
                source_index_name=source_index_name,
                target_index_name=target_index_name,
                merge_policy=index_service_pb.MergePolicy(**(merge_policy or {})),
                **target_index_engine,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def delete_consumer(
            self,
            consumer_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> consumer_service_pb.DeleteConsumerResponse:
        """
        Delete consumer by index and consumer names

        Args:
            consumer_name: consumer name
            request_id: request id
            session_id: session id
        """
        return await self.stubs['consumer_api'].delete_consumer(
            consumer_service_pb.DeleteConsumerRequest(consumer_name=consumer_name),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def delete_documents(
            self,
            index_name: str,
            query: query_pb.Query,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.IndexDocumentResponse:
        """
        Delete document with primary key

        Args:
            index_name: index name
            query: query used for retrieving documents for deletion
            request_id:
            session_id:
        """
        return await self.stubs['index_api'].delete_documents(
            index_service_pb.DeleteDocumentsRequest(
                index_name=index_name,
                query=query,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def delete_index(
            self,
            index_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.DeleteIndexResponse:
        """
        Delete index

        Args:
            index_name: index name
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].delete_index(
            index_service_pb.DeleteIndexRequest(index_name=index_name),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def get_consumer(
            self,
            consumer_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> consumer_service_pb.GetConsumerResponse:
        """
        Consumer metadata

        Args:
            consumer_name: consumer name
            request_id: request id
            session_id: session id
        Return:
            Consumer description
        """
        return await self.stubs['consumer_api'].get_consumer(
            consumer_service_pb.GetConsumerRequest(consumer_name=consumer_name),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def get_consumers(
            self,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> consumer_service_pb.GetConsumersResponse:
        """
        All active consumers

        Args:
            request_id: request id
            session_id: session id
        Returns:
            Consumers list
        """
        return await self.stubs['consumer_api'].get_consumers(
            consumer_service_pb.GetConsumersRequest(),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def get_index(
            self,
            index_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.GetIndexResponse:
        """
        Index metadata

        Args:
            index_name: index name
            request_id: request id
            session_id: session id
        Returns:
            Index description
        """
        return await self.stubs['index_api'].get_index(
            index_service_pb.GetIndexRequest(index_name=index_name),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def get_indices(
            self,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.GetIndicesResponse:
        """
        All indices metadata

        Args:
            request_id: request id
            session_id: session id
        Returns:
            Indices list
        """
        return await self.stubs['index_api'].get_indices(
            index_service_pb.GetIndicesRequest(),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def get_indices_aliases(
            self,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.GetIndicesAliasesResponse:
        """
        Get all aliases for all indices

        Args:
            request_id: request id
            session_id: session id
        Returns:
            Aliases list
        """
        return await self.stubs['index_api'].get_indices_aliases(
            index_service_pb.GetIndicesAliasesRequest(),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def documents(
            self,
            index_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Retrieve all documents from the index

        Args:
            index_name: index name
            request_id: request id
            session_id: session id
        """
        # asyncfor is buggy: https://github.com/grpc/grpc/issues/32005
        streaming_call = self.stubs['index_api'].documents(
            index_service_pb.DocumentsRequest(index_name=index_name),
            metadata=setup_metadata(session_id, request_id),
        )
        while True:
            document = await asyncio.create_task(streaming_call.read())
            if document == grpc.aio.EOF:
                break
            yield document.document

    @expose
    async def index_document_stream(
            self,
            index_name: str,
            documents: Union[Iterable[str], str] = None,
            conflict_strategy: Optional[str] = None,
            bulk_size: int = 100,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.IndexDocumentStreamResponse:
        """
        Index documents bulky

        Args:
            index_name: index name
            documents: list of bytes
            conflict_strategy: recommended to set to DoNothing for large updates and maintain uniqueness in your application
            bulk_size: document portion size to send
            request_id: request id
            session_id: session id
        """
        if isinstance(conflict_strategy, str):
            conflict_strategy = index_service_pb.ConflictStrategy.Value(conflict_strategy)

        if documents is None and not sys.stdin.isatty():
            def documents_iter():
                for line in sys.stdin:
                    yield line.strip().encode()

            documents = documents_iter()

        return await self.stubs['index_api'].index_document_stream(
            documents_portion_iter(index_name, documents, bulk_size, conflict_strategy=conflict_strategy),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def index_document(
            self,
            index_name: str,
            document: Union[dict, bytes, str],
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.IndexDocumentResponse:
        """
        Index document

        Args:
            index_name: index name
            document: bytes
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].index_document(
            index_service_pb.IndexDocumentRequest(
                index_name=index_name,
                document=json.dumps(document) if isinstance(document, dict) else document,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def search(
            self,
            search_request: dict,
            ignore_not_found: bool = False,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> query_pb.SearchResponse:
        """Send search request. `Query` object can be created manually or by using `aiosumma.parser` module.

        Args:
            search_request:
            ignore_not_found: do not raise `StatusCode.NOT_FOUND` and return empty SearchResponse
            request_id: request id
            session_id: session id
        """
        try:
            if isinstance(search_request, Dict):
                dict_search_request = search_request
                search_request = search_service_pb.SearchRequest()
                ParseDict(dict_search_request, search_request)
            return await self.stubs['search_api'].search(
                search_request,
                metadata=setup_metadata(session_id, request_id),
            )
        except AioRpcError as e:
            if ignore_not_found and e.code() == StatusCode.NOT_FOUND:
                return query_pb.SearchResponse()
            raise

    @expose
    async def search_documents(
            self,
            search_request: dict,
            tags: Optional[Dict[str, str]] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> List[dict]:
        """Send search request and interprets first collector as, thus parse returned documents as json.

        Args:
            search_request: index queries
            tags: extra dict for logging purposes
            request_id: request id
            session_id: session id
        """
        search_results = await self.search(search_request=search_request, tags=tags, request_id=request_id, session_id=session_id)
        return [
            json.loads(scored_document.document)
            for scored_document in search_results.collector_outputs[0].documents.scored_documents
        ]

    @expose
    async def merge_segments(
            self,
            index_name: str,
            segment_ids: List[str],
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.MergeSegmentsResponse:
        """
        Merge a list of segments into a single one

        Args:
            index_name: index name
            segment_ids: segment ids
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].merge_segments(
            index_service_pb.MergeSegmentsRequest(index_name=index_name, segment_ids=segment_ids),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def set_index_alias(
            self,
            index_alias: str,
            index_name: str,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.SetIndexAliasResponse:
        """
        Set or reassign the alias for an index

        Args:
            index_alias: index alias
            index_name: index name
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].set_index_alias(
            index_service_pb.SetIndexAliasRequest(index_alias=index_alias, index_name=index_name),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def vacuum_index(
            self,
            index_name: str,
            excluded_segments: Optional[List[str]] = None,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.VacuumIndexResponse:
        """
        Vacuuming index. It cleans every segment from deleted documents, one by one.

        Args:
            index_name: index name
            excluded_segments: segments that must not be merged
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].vacuum_index(
            index_service_pb.VacuumIndexRequest(index_name=index_name, excluded_segments=excluded_segments),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def warmup_index(
            self,
            index_name: str,
            is_full: bool = False,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.VacuumIndexResponse:
        """
        Warm index up. It loads all hot parts or index into memory and makes further first queries to the index faster.

        Args:
            index_name: index name
            is_full: should full or partial warm up be done
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].warmup_index(
            index_service_pb.WarmupIndexRequest(index_name=index_name, is_full=is_full),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose
    async def get_top_terms(
            self,
            index_name: str,
            field_name: str,
            top_k: int,
            request_id: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> index_service_pb.VacuumIndexResponse:
        """
        Get top terms by the number for the index

        Args:
            index_name: index name
            field_name: field name
            top_k: extract top-K terms
            request_id: request id
            session_id: session id
        """
        return await self.stubs['reflection_api'].get_top_terms(
            reflection_service_pb.GetTopTermsRequest(
                index_name=index_name,
                field_name=field_name,
                top_k=top_k,
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    async def get_one_by_field_value(self, index_alias, field, value):
        response = await self.search({
            'index_alias': index_alias,
            'query': {'term': {'field': field, 'value': value}},
            'collectors': [{'top_docs': {'limit': 1}}],
        })
        if response.collector_outputs[0].documents.scored_documents:
            return json.loads(response.collector_outputs[0].documents.scored_documents[0].document)

