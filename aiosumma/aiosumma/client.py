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

from .proto import consumer_service_pb2 as consumer_service_pb
from .proto import index_service_pb2 as index_service_pb
from .proto import query_pb2 as query_pb
from .proto import reflection_service_pb2 as reflection_service_pb
from .proto import search_service_pb2 as search_service_pb
from .proto.consumer_service_pb2_grpc import ConsumerApiStub
from .proto.index_service_pb2_grpc import IndexApiStub
from .proto.reflection_service_pb2_grpc import ReflectionApiStub
from .proto.search_service_pb2_grpc import SearchApiStub
from .proto.utils_pb2 import (  # noqa
    Asc,
    Desc,
)


def setup_metadata(session_id, request_id):
    metadata = []
    if session_id:
        metadata.append(('session-id', session_id))
    if request_id:
        metadata.append(('request-id', request_id))
    return metadata


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
        file: Optional[Union[index_service_pb.AttachFileEngineRequest, Dict]] = None,
        remote: Optional[Union[index_service_pb.AttachRemoteEngineRequest, Dict]] = None,
        ipfs: Optional[Union[index_service_pb.AttachIpfsEngineRequest, Dict]] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.AttachIndexResponse:
        """
        Attach index to Summa. It must be placed under data directory named as `index_name`

        Args:
            index_name: index name
            file: attaching index placed under `<data_path>/<index_name>` directory
            remote: attaching remote index
            ipfs: attaching ipfs index
            request_id: request id
            session_id: session id
        """
        request = {'index_name': index_name}
        if file:
            request['attach_file_engine_request'] = file
        elif remote:
            request['attach_remote_engine_request'] = remote
        elif ipfs:
            request['attach_ipfs_engine_request'] = ipfs
        return await self.stubs['index_api'].attach_index(
            index_service_pb.AttachIndexRequest(**request),
            metadata=setup_metadata(session_id, request_id),
        )

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
        index_engine: str,
        schema: str,
        compression: Optional[Union[str, int]] = None,
        blocksize: Optional[int] = None,
        sort_by_field: Optional[Tuple] = None,
        index_attributes: Optional[Dict] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.CreateIndexResponse:
        """
        Create index

        Args:
            index_name: index name
            index_engine: "File" or "Memory"
            schema: Tantivy index schema
            compression: Tantivy index compression
            blocksize: Docstore blocksize
            index_attributes: Various index attributes such as default fields, multi fields, primary keys etc.
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
                index_engine=index_engine,
                schema=schema,
                compression=compression,
                blocksize=blocksize,
                index_attributes=index_attributes,
                sort_by_field=index_service_pb.SortByField(
                    field=sort_by_field[0],
                    order=sort_by_field[1],
                ) if sort_by_field else None
            ),
            metadata=setup_metadata(session_id, request_id),
        )

    @expose(with_from_file=True)
    async def migrate_index(
        self,
        source_index_name: str,
        target_index_name: str,
        target_index_engine: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> index_service_pb.CreateIndexResponse:
        """
        Migrates existing index to new engine

        Args:
            source_index_name: source index name
            target_index_name: target index name
            target_index_engine: "File", "Memory" or "Ipfs"
            request_id: request id
            session_id: session id
        """
        return await self.stubs['index_api'].migrate_index(
            index_service_pb.MigrateIndexRequest(
                source_index_name=source_index_name,
                target_index_name=target_index_name,
                target_index_engine=target_index_engine,
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
    async def delete_document(
        self,
        index_alias: str,
        primary_key: index_service_pb.PrimaryKey,
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
            metadata=setup_metadata(session_id, request_id),
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
            metadata=setup_metadata(session_id, request_id),
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
    ) -> query_pb.SearchResponse:
        """Send search request. `Query` object can be created manually or by using `aiosumma.parser` module.

        Args:
            index_alias: index alias
            query: parsed `Query`
            collectors: query_pb.Collector list
            tags: extra dict for logging purposes
            ignore_not_found: do not raise `StatusCode.NOT_FOUND` and return empty SearchResponse
            request_id: request id
            session_id: session id
        """
        if isinstance(collectors, (Dict, query_pb.Collector)):
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
                    collector = query_pb.Collector()
                    ParseDict(dict_collector, collector)
                search_request.collectors.append(collector)
            return await self.stubs['search_api'].search(
                search_request,
                metadata=setup_metadata(session_id, request_id),
            )
        except AioRpcError as e:
            if ignore_not_found and e.code() == StatusCode.NOT_FOUND:
                return query_pb.SearchResponse()
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
            metadata=setup_metadata(session_id, request_id),
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
            metadata=setup_metadata(session_id, request_id),
        )
