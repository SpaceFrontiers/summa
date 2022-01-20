import query_pb2 as _query_pb2
import utils_pb2 as _utils_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

Async: CommitMode
Brotli: Compression
DESCRIPTOR: _descriptor.FileDescriptor
File: CreateIndexEngineRequest
Ipfs: CreateIndexEngineRequest
Lz4: Compression
Memory: CreateIndexEngineRequest
None: Compression
Snappy: Compression
Sync: CommitMode
Zstd: Compression

class AttachFileEngineRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class AttachIndexRequest(_message.Message):
    __slots__ = ["attach_file_engine_request", "attach_ipfs_engine_request", "attach_remote_engine_request", "index_name"]
    ATTACH_FILE_ENGINE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    ATTACH_IPFS_ENGINE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    ATTACH_REMOTE_ENGINE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    attach_file_engine_request: AttachFileEngineRequest
    attach_ipfs_engine_request: AttachIpfsEngineRequest
    attach_remote_engine_request: AttachRemoteEngineRequest
    index_name: str
    def __init__(self, index_name: _Optional[str] = ..., attach_file_engine_request: _Optional[_Union[AttachFileEngineRequest, _Mapping]] = ..., attach_remote_engine_request: _Optional[_Union[AttachRemoteEngineRequest, _Mapping]] = ..., attach_ipfs_engine_request: _Optional[_Union[AttachIpfsEngineRequest, _Mapping]] = ...) -> None: ...

class AttachIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class AttachIpfsEngineRequest(_message.Message):
    __slots__ = ["cid"]
    CID_FIELD_NUMBER: _ClassVar[int]
    cid: str
    def __init__(self, cid: _Optional[str] = ...) -> None: ...

class AttachRemoteEngineRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ChunkedCacheConfig(_message.Message):
    __slots__ = ["cache_size", "chunk_size"]
    CACHE_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    cache_size: int
    chunk_size: int
    def __init__(self, chunk_size: _Optional[int] = ..., cache_size: _Optional[int] = ...) -> None: ...

class CommitIndexRequest(_message.Message):
    __slots__ = ["commit_mode", "index_alias"]
    COMMIT_MODE_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    commit_mode: CommitMode
    index_alias: str
    def __init__(self, index_alias: _Optional[str] = ..., commit_mode: _Optional[_Union[CommitMode, str]] = ...) -> None: ...

class CommitIndexResponse(_message.Message):
    __slots__ = ["elapsed_secs"]
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    def __init__(self, elapsed_secs: _Optional[float] = ...) -> None: ...

class CreateIndexRequest(_message.Message):
    __slots__ = ["blocksize", "compression", "index_attributes", "index_engine", "index_name", "schema", "sort_by_field"]
    BLOCKSIZE_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    INDEX_ENGINE_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_FIELD_NUMBER: _ClassVar[int]
    blocksize: int
    compression: Compression
    index_attributes: IndexAttributes
    index_engine: CreateIndexEngineRequest
    index_name: str
    schema: str
    sort_by_field: SortByField
    def __init__(self, index_name: _Optional[str] = ..., index_engine: _Optional[_Union[CreateIndexEngineRequest, str]] = ..., schema: _Optional[str] = ..., compression: _Optional[_Union[Compression, str]] = ..., blocksize: _Optional[int] = ..., sort_by_field: _Optional[_Union[SortByField, _Mapping]] = ..., index_attributes: _Optional[_Union[IndexAttributes, _Mapping]] = ...) -> None: ...

class CreateIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class DeleteDocumentsRequest(_message.Message):
    __slots__ = ["index_alias", "query"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    query: _query_pb2.Query
    def __init__(self, index_alias: _Optional[str] = ..., query: _Optional[_Union[_query_pb2.Query, _Mapping]] = ...) -> None: ...

class DeleteDocumentsResponse(_message.Message):
    __slots__ = ["deleted_documents"]
    DELETED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    deleted_documents: int
    def __init__(self, deleted_documents: _Optional[int] = ...) -> None: ...

class DeleteIndexRequest(_message.Message):
    __slots__ = ["index_name"]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class DeleteIndexResponse(_message.Message):
    __slots__ = ["index_name"]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class FileEngineConfig(_message.Message):
    __slots__ = ["path"]
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class GetIndexRequest(_message.Message):
    __slots__ = ["index_alias"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    def __init__(self, index_alias: _Optional[str] = ...) -> None: ...

class GetIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class GetIndicesAliasesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetIndicesAliasesResponse(_message.Message):
    __slots__ = ["indices_aliases"]
    class IndicesAliasesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INDICES_ALIASES_FIELD_NUMBER: _ClassVar[int]
    indices_aliases: _containers.ScalarMap[str, str]
    def __init__(self, indices_aliases: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GetIndicesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetIndicesResponse(_message.Message):
    __slots__ = ["indices"]
    INDICES_FIELD_NUMBER: _ClassVar[int]
    indices: _containers.RepeatedCompositeFieldContainer[IndexDescription]
    def __init__(self, indices: _Optional[_Iterable[_Union[IndexDescription, _Mapping]]] = ...) -> None: ...

class IndexAttributes(_message.Message):
    __slots__ = ["created_at", "default_fields", "default_index_name", "default_snippets", "description", "multi_fields", "unique_fields"]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_SNIPPETS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    MULTI_FIELDS_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_FIELDS_FIELD_NUMBER: _ClassVar[int]
    created_at: int
    default_fields: _containers.RepeatedScalarFieldContainer[str]
    default_index_name: str
    default_snippets: _containers.RepeatedScalarFieldContainer[str]
    description: str
    multi_fields: _containers.RepeatedScalarFieldContainer[str]
    unique_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, created_at: _Optional[int] = ..., unique_fields: _Optional[_Iterable[str]] = ..., default_fields: _Optional[_Iterable[str]] = ..., multi_fields: _Optional[_Iterable[str]] = ..., default_index_name: _Optional[str] = ..., description: _Optional[str] = ..., default_snippets: _Optional[_Iterable[str]] = ...) -> None: ...

class IndexDescription(_message.Message):
    __slots__ = ["compression", "index_aliases", "index_engine", "index_name", "num_docs"]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIASES_FIELD_NUMBER: _ClassVar[int]
    INDEX_ENGINE_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_DOCS_FIELD_NUMBER: _ClassVar[int]
    compression: Compression
    index_aliases: _containers.RepeatedScalarFieldContainer[str]
    index_engine: IndexEngineConfig
    index_name: str
    num_docs: int
    def __init__(self, index_name: _Optional[str] = ..., index_aliases: _Optional[_Iterable[str]] = ..., index_engine: _Optional[_Union[IndexEngineConfig, _Mapping]] = ..., num_docs: _Optional[int] = ..., compression: _Optional[_Union[Compression, str]] = ...) -> None: ...

class IndexDocumentOperation(_message.Message):
    __slots__ = ["document"]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    document: bytes
    def __init__(self, document: _Optional[bytes] = ...) -> None: ...

class IndexDocumentRequest(_message.Message):
    __slots__ = ["document", "index_alias"]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    document: bytes
    index_alias: str
    def __init__(self, index_alias: _Optional[str] = ..., document: _Optional[bytes] = ...) -> None: ...

class IndexDocumentResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class IndexDocumentStreamRequest(_message.Message):
    __slots__ = ["documents", "index_alias"]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedScalarFieldContainer[bytes]
    index_alias: str
    def __init__(self, index_alias: _Optional[str] = ..., documents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class IndexDocumentStreamResponse(_message.Message):
    __slots__ = ["elapsed_secs", "failed_docs", "success_docs"]
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    FAILED_DOCS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_DOCS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    failed_docs: int
    success_docs: int
    def __init__(self, success_docs: _Optional[int] = ..., failed_docs: _Optional[int] = ..., elapsed_secs: _Optional[float] = ...) -> None: ...

class IndexEngineConfig(_message.Message):
    __slots__ = ["file", "ipfs", "memory", "remote"]
    FILE_FIELD_NUMBER: _ClassVar[int]
    IPFS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    REMOTE_FIELD_NUMBER: _ClassVar[int]
    file: FileEngineConfig
    ipfs: IpfsEngineConfig
    memory: MemoryEngineConfig
    remote: RemoteEngineConfig
    def __init__(self, file: _Optional[_Union[FileEngineConfig, _Mapping]] = ..., memory: _Optional[_Union[MemoryEngineConfig, _Mapping]] = ..., remote: _Optional[_Union[RemoteEngineConfig, _Mapping]] = ..., ipfs: _Optional[_Union[IpfsEngineConfig, _Mapping]] = ...) -> None: ...

class IndexOperation(_message.Message):
    __slots__ = ["index_document"]
    INDEX_DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    index_document: IndexDocumentOperation
    def __init__(self, index_document: _Optional[_Union[IndexDocumentOperation, _Mapping]] = ...) -> None: ...

class IpfsEngineConfig(_message.Message):
    __slots__ = ["chunked_cache_config", "cid", "path"]
    CHUNKED_CACHE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    chunked_cache_config: ChunkedCacheConfig
    cid: str
    path: str
    def __init__(self, cid: _Optional[str] = ..., chunked_cache_config: _Optional[_Union[ChunkedCacheConfig, _Mapping]] = ..., path: _Optional[str] = ...) -> None: ...

class MemoryEngineConfig(_message.Message):
    __slots__ = ["schema"]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: str
    def __init__(self, schema: _Optional[str] = ...) -> None: ...

class MergeSegmentsRequest(_message.Message):
    __slots__ = ["index_alias", "segment_ids"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    segment_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, index_alias: _Optional[str] = ..., segment_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class MergeSegmentsResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class MigrateIndexRequest(_message.Message):
    __slots__ = ["source_index_name", "target_index_engine", "target_index_name"]
    SOURCE_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_INDEX_ENGINE_FIELD_NUMBER: _ClassVar[int]
    TARGET_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    source_index_name: str
    target_index_engine: CreateIndexEngineRequest
    target_index_name: str
    def __init__(self, source_index_name: _Optional[str] = ..., target_index_name: _Optional[str] = ..., target_index_engine: _Optional[_Union[CreateIndexEngineRequest, str]] = ...) -> None: ...

class MigrateIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class PrimaryKey(_message.Message):
    __slots__ = ["i64", "str"]
    I64_FIELD_NUMBER: _ClassVar[int]
    STR_FIELD_NUMBER: _ClassVar[int]
    i64: int
    str: str
    def __init__(self, str: _Optional[str] = ..., i64: _Optional[int] = ...) -> None: ...

class RemoteEngineConfig(_message.Message):
    __slots__ = ["chunked_cache_config", "headers_template", "method", "url_template"]
    class HeadersTemplateEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CHUNKED_CACHE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    HEADERS_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    URL_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    chunked_cache_config: ChunkedCacheConfig
    headers_template: _containers.ScalarMap[str, str]
    method: str
    url_template: str
    def __init__(self, method: _Optional[str] = ..., url_template: _Optional[str] = ..., headers_template: _Optional[_Mapping[str, str]] = ..., chunked_cache_config: _Optional[_Union[ChunkedCacheConfig, _Mapping]] = ...) -> None: ...

class SetIndexAliasRequest(_message.Message):
    __slots__ = ["index_alias", "index_name"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    index_name: str
    def __init__(self, index_alias: _Optional[str] = ..., index_name: _Optional[str] = ...) -> None: ...

class SetIndexAliasResponse(_message.Message):
    __slots__ = ["old_index_name"]
    OLD_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    old_index_name: str
    def __init__(self, old_index_name: _Optional[str] = ...) -> None: ...

class SortByField(_message.Message):
    __slots__ = ["field", "order"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    field: str
    order: _utils_pb2.Order
    def __init__(self, field: _Optional[str] = ..., order: _Optional[_Union[_utils_pb2.Order, str]] = ...) -> None: ...

class VacuumIndexRequest(_message.Message):
    __slots__ = ["index_alias"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    def __init__(self, index_alias: _Optional[str] = ...) -> None: ...

class VacuumIndexResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class WarmupIndexRequest(_message.Message):
    __slots__ = ["index_alias", "is_full"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    IS_FULL_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    is_full: bool
    def __init__(self, index_alias: _Optional[str] = ..., is_full: bool = ...) -> None: ...

class WarmupIndexResponse(_message.Message):
    __slots__ = ["elapsed_secs"]
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    def __init__(self, elapsed_secs: _Optional[float] = ...) -> None: ...

class CreateIndexEngineRequest(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Compression(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class CommitMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
