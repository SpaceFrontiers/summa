import sys
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import orjson as json
from aiogrpcclient import (
    BaseGrpcClient,
    expose,
)
from grpc import StatusCode
from grpc.experimental.aio import AioRpcError
from izihawa_utils.pb_to_json import ParseDict
from summa.proto import consumer_service_pb2 as consumer_service_pb
from summa.proto import index_service_pb2 as index_service_pb
from summa.proto import reflection_service_pb2 as reflection_service_pb
from summa.proto import search_service_pb2 as search_service_pb
from summa.proto.consumer_service_pb2_grpc import ConsumerApiStub
from summa.proto.index_service_pb2_grpc import IndexApiStub
from summa.proto.reflection_service_pb2_grpc import ReflectionApiStub
from summa.proto.search_service_pb2_grpc import SearchApiStub
from summa.proto.utils_pb2 import (  # noqa
    Asc,
    Desc,
)


class SummaClient(BaseGrpcClient):
    stub_clses = {
        'consumer_api': ConsumerApiStub,
        'index_api': IndexApiStub,
        'reflection_api': ReflectionApiStub,
        'search_api': SearchApiStub,
    }

    @expose
    async def commit_index(
        self,
        index_alias: str,
        commit_mode: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.CommitIndexResponse:
        """
        Commit index asynchronously. A commit will be scheduled and be done eventually.
        Executing commit means stopping all consumption before commit and starting it again
        after.

        Args:
            index_alias: index alias
            commit_mode: Sync | Async
            request_id: request id
            session_id: session id
        Returns:
            Commit scheduling result
        """
        return await self.stubs['index_api'].commit_index(
            index_service_pb.CommitIndexRequest(
                index_alias=index_alias,
                commit_mode=commit_mode,
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def create_consumer(
        self,
        index_alias: str,
        consumer_name: str,
        bootstrap_servers: List[str],
        group_id: str,
        topics: List[str],
        threads: int = None,
        request_id: str = None,
        session_id: str = None,
    ) -> consumer_service_pb.CreateConsumerResponse:
        """
        Create consumer and corresponding topics in Kafka.
        The newly created consumer starts immediately after creation

        Args:
            index_alias: index alias
            consumer_name: consumer name that will be used for topic creation in Kafka too
            bootstrap_servers: list of bootstrap servers
            group_id: group_id for Kafka topic consumption
            topics: list of topics
            threads: number of threads to read topics and number of partitions in Kafka topic
            request_id: request id
            session_id: session id
        """
        return await self.stubs['consumer_api'].create_consumer(
            consumer_service_pb.CreateConsumerRequest(
                consumer_name=consumer_name,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                index_alias=index_alias,
                topics=topics,
                threads=threads,
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def alter_index(
        self,
        index_name: str,
        compression: Optional[str] = None,
        sort_by_field: Optional[Tuple] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.CreateIndexResponse:
        """
        Alter index options like compression and ordering

        Args:
            index_name: index name
            compression: Tantivy index compression
            sort_by_field: (field_name, order)
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].alter_index(
            index_service_pb.AlterIndexRequest(
                index_name=index_name,
                compression=index_service_pb.Compression.Value(compression) if compression is not None else None,
                sort_by_field=index_service_pb.SortByField(
                    field=sort_by_field[0],
                    order=sort_by_field[1],
                ) if sort_by_field else None
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose(with_from_file=True)
    async def create_index(
        self,
        index_name: str,
        fields: str,
        primary_key: Optional[str] = None,
        default_fields: Optional[List[str]] = None,
        multi_fields: Optional[List[str]] = None,
        stop_words: Optional[List[str]] = None,
        compression: Optional[Union[str, int]] = None,
        writer_heap_size_bytes: Optional[int] = None,
        writer_threads: Optional[int] = None,
        autocommit_interval_ms: Optional[int] = None,
        sort_by_field: Optional[Tuple] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.CreateIndexResponse:
        """
        Create index

        Args:
            index_name: index name
            fields: Tantivy index schema
            primary_key: primary key is used during insertion to check duplicates
            default_fields: fields that are used to search by default
            multi_fields: fields that can have multiple values
            stop_words: list of words that won't be parsed
            compression: Tantivy index compression
            writer_heap_size_bytes: Tantivy writer heap size in bytes, shared between all threads
            writer_threads: Tantivy writer threads
            autocommit_interval_ms: if true then there will be a separate thread committing index every nth milliseconds
                set by this parameter
            request_id: request id
            session_id: session id
            sort_by_field: (field_name, order)
        """
        if isinstance(compression, str):
            compression = index_service_pb.Compression.Value(compression)
        elif isinstance(compression, int):
            compression = index_service_pb.Compression.Name(compression)
        return await self.stubs['index_api'].create_index(
            index_service_pb.CreateIndexRequest(
                index_name=index_name,
                fields=fields,
                primary_key=primary_key,
                default_fields=default_fields,
                multi_fields=multi_fields,
                stop_words=stop_words,
                compression=compression,
                writer_heap_size_bytes=writer_heap_size_bytes,
                writer_threads=writer_threads,
                autocommit_interval_ms=autocommit_interval_ms,
                sort_by_field=index_service_pb.SortByField(
                    field=sort_by_field[0],
                    order=sort_by_field[1],
                ) if sort_by_field else None
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def delete_consumer(
        self,
        index_alias: str,
        consumer_name: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> consumer_service_pb.DeleteConsumerResponse:
        """
        Delete consumer by index and consumer names

        Args:
            index_alias: index alias
            consumer_name: consumer name
            request_id: request id
            session_id: session id
        """
        return await self.stubs['consumer_api'].delete_consumer(
            consumer_service_pb.DeleteConsumerRequest(index_alias=index_alias, consumer_name=consumer_name),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def delete_document(
        self,
        index_alias: str,
        primary_key: int,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.IndexDocumentResponse:
        """
        Delete document with primary key

        Args:
            index_alias: index alias
            primary_key: bytes
            request_id:
            session_id:
        """
        return await self.stubs['index_api'].delete_document(
            index_service_pb.DeleteDocumentRequest(
                index_alias=index_alias,
                primary_key=primary_key,
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
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
            metadata=(('request-id', request_id), ('session-id', session_id)),
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
            metadata=(('request-id', request_id), ('session-id', session_id)),
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
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def get_index(
        self,
        index_alias: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.GetIndexResponse:
        """
        Index metadata

        Args:
            index_alias: index alias
            request_id: request id
            session_id: session id
        Returns:
            Index description
        """
        return await self.stubs['index_api'].get_index(
            index_service_pb.GetIndexRequest(index_alias=index_alias),
            metadata=(('request-id', request_id), ('session-id', session_id)),
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
            metadata=(('request-id', request_id), ('session-id', session_id)),
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
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def index_document_stream(
        self,
        index_alias: str,
        documents: Union[Iterable[str], str] = None,
        bulk_size: int = 1000,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.IndexDocumentStreamResponse:
        """
        Index documents bulky

        Args:
            index_alias: index alias
            documents: list of bytes
            bulk_size: document portion size to send
            request_id: request id
            session_id: session id
        """
        if documents is None and not sys.stdin.isatty():
            def documents_iter():
                for line in sys.stdin:
                    yield line.strip().encode()
            documents = documents_iter()

        def documents_portion_iter():
            documents_portion = []
            for document in documents:
                documents_portion.append(document)
                if len(documents_portion) > bulk_size:
                    yield index_service_pb.IndexDocumentStreamRequest(
                        index_alias=index_alias,
                        documents=documents_portion,
                    )
                    documents_portion = []
            if documents_portion:
                yield index_service_pb.IndexDocumentStreamRequest(
                    index_alias=index_alias,
                    documents=documents_portion,
                )

        return await self.stubs['index_api'].index_document_stream(
            documents_portion_iter(),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def index_document(
        self,
        index_alias: str,
        document: Union[dict, bytes, str],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.IndexDocumentResponse:
        """
        Index document

        Args:
            index_alias: index alias
            document: bytes
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].index_document(
            index_service_pb.IndexDocumentRequest(
                index_alias=index_alias,
                document=json.dumps(document) if isinstance(document, dict) else document,
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def search(
        self,
        index_alias: str,
        query: dict,
        collectors: Union[dict, List[dict]],
        tags: Optional[Dict[str, str]] = None,
        ignore_not_found: bool = False,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> search_service_pb.SearchResponse:
        """pu
        Send search request. `Query` object can be created manually or by using `aiosumma.parser` module.

        Args:
            index_alias: index alias
            query: parsed `Query`
            collectors: search_service_pb.Collector list
            tags: extra dict for logging purposes
            ignore_not_found: do not raise `StatusCode.NOT_FOUND` and return empty SearchResponse
            request_id: request id
            session_id: session id
        """
        if isinstance(collectors, (Dict, search_service_pb.Collector)):
            collectors = [collectors]

        try:
            search_request = search_service_pb.SearchRequest(
                index_alias=index_alias,
                query=query,
                tags=tags,
            )
            for collector in collectors:
                if isinstance(collector, Dict):
                    dict_collector = collector
                    collector = search_service_pb.Collector()
                    ParseDict(dict_collector, collector)
                search_request.collectors.append(collector)
            return await self.stubs['search_api'].search(
                search_request,
                metadata=(('request-id', request_id), ('session-id', session_id)),
            )
        except AioRpcError as e:
            if ignore_not_found and e.code() == StatusCode.NOT_FOUND:
                return search_service_pb.SearchResponse()
            raise

    @expose
    async def merge_segments(
        self,
        index_alias: str,
        segment_ids: List[str],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.MergeSegmentsResponse:
        """
        Merge a list of segments into a single one

        Args:
            index_alias: index alias
            segment_ids: segment ids
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].merge_segments(
            index_service_pb.MergeSegmentsRequest(index_alias=index_alias, segment_ids=segment_ids),
            metadata=(('request-id', request_id), ('session-id', session_id)),
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
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def vacuum_index(
        self,
        index_alias: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.VacuumIndexResponse:
        """
        Vacuuming index. It cleans every segment from deleted documents, one by one.

        Args:
            index_alias: index alias
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].vacuum_index(
            index_service_pb.VacuumIndexRequest(index_alias=index_alias),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )

    @expose
    async def get_top_terms(
        self,
        index_alias: str,
        field_name: str,
        top_k: int,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.VacuumIndexResponse:
        """
        Get top terms by the number for the index

        Args:
            index_alias: index alias
            field_name: field name
            top_k: extract top-K terms
            request_id: request id
            session_id: session id
        """
        return await self.stubs['reflection_api'].get_top_terms(
            reflection_service_pb.GetTopTermsRequest(
                index_alias=index_alias,
                field_name=field_name,
                top_k=top_k,
            ),
            metadata=(('request-id', request_id), ('session-id', session_id)),
        )
