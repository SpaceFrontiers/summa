import query_pb2 as _query_pb2
import utils_pb2 as _utils_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ConflictStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DO_NOTHING: _ClassVar[ConflictStrategy]
    OVERWRITE_ALWAYS: _ClassVar[ConflictStrategy]
    OVERWRITE: _ClassVar[ConflictStrategy]
    MERGE: _ClassVar[ConflictStrategy]

class Compression(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    None: _ClassVar[Compression]
    Zstd: _ClassVar[Compression]
    Zstd7: _ClassVar[Compression]
    Zstd9: _ClassVar[Compression]
    Zstd14: _ClassVar[Compression]
    Zstd19: _ClassVar[Compression]
    Zstd22: _ClassVar[Compression]
DO_NOTHING: ConflictStrategy
OVERWRITE_ALWAYS: ConflictStrategy
OVERWRITE: ConflictStrategy
MERGE: ConflictStrategy
None: Compression
Zstd: Compression
Zstd7: Compression
Zstd9: Compression
Zstd14: Compression
Zstd19: Compression
Zstd22: Compression

class MergePolicy(_message.Message):
    __slots__ = ("log", "temporal")
    LOG_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    log: LogMergePolicy
    temporal: TemporalMergePolicy
    def __init__(self, log: _Optional[_Union[LogMergePolicy, _Mapping]] = ..., temporal: _Optional[_Union[TemporalMergePolicy, _Mapping]] = ...) -> None: ...

class AttachFileEngineRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class AttachRemoteEngineRequest(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: RemoteEngineConfig
    def __init__(self, config: _Optional[_Union[RemoteEngineConfig, _Mapping]] = ...) -> None: ...

class AttachIndexRequest(_message.Message):
    __slots__ = ("index_name", "file", "remote", "merge_policy", "query_parser_config")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    REMOTE_FIELD_NUMBER: _ClassVar[int]
    MERGE_POLICY_FIELD_NUMBER: _ClassVar[int]
    QUERY_PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    file: AttachFileEngineRequest
    remote: AttachRemoteEngineRequest
    merge_policy: MergePolicy
    query_parser_config: _query_pb2.QueryParserConfig
    def __init__(self, index_name: _Optional[str] = ..., file: _Optional[_Union[AttachFileEngineRequest, _Mapping]] = ..., remote: _Optional[_Union[AttachRemoteEngineRequest, _Mapping]] = ..., merge_policy: _Optional[_Union[MergePolicy, _Mapping]] = ..., query_parser_config: _Optional[_Union[_query_pb2.QueryParserConfig, _Mapping]] = ...) -> None: ...

class AttachIndexResponse(_message.Message):
    __slots__ = ("index",)
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class CommitIndexRequest(_message.Message):
    __slots__ = ("index_name", "with_hotcache")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    WITH_HOTCACHE_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    with_hotcache: bool
    def __init__(self, index_name: _Optional[str] = ..., with_hotcache: bool = ...) -> None: ...

class CommitIndexResponse(_message.Message):
    __slots__ = ("elapsed_secs",)
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    def __init__(self, elapsed_secs: _Optional[float] = ...) -> None: ...

class CopyDocumentsRequest(_message.Message):
    __slots__ = ("source_index_name", "target_index_name", "conflict_strategy")
    SOURCE_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    source_index_name: str
    target_index_name: str
    conflict_strategy: ConflictStrategy
    def __init__(self, source_index_name: _Optional[str] = ..., target_index_name: _Optional[str] = ..., conflict_strategy: _Optional[_Union[ConflictStrategy, str]] = ...) -> None: ...

class CopyDocumentsResponse(_message.Message):
    __slots__ = ("elapsed_secs", "copied_documents")
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    COPIED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    copied_documents: int
    def __init__(self, elapsed_secs: _Optional[float] = ..., copied_documents: _Optional[int] = ...) -> None: ...

class CopyIndexRequest(_message.Message):
    __slots__ = ("source_index_name", "target_index_name", "file", "memory", "merge_policy")
    SOURCE_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    MERGE_POLICY_FIELD_NUMBER: _ClassVar[int]
    source_index_name: str
    target_index_name: str
    file: CreateFileEngineRequest
    memory: CreateMemoryEngineRequest
    merge_policy: MergePolicy
    def __init__(self, source_index_name: _Optional[str] = ..., target_index_name: _Optional[str] = ..., file: _Optional[_Union[CreateFileEngineRequest, _Mapping]] = ..., memory: _Optional[_Union[CreateMemoryEngineRequest, _Mapping]] = ..., merge_policy: _Optional[_Union[MergePolicy, _Mapping]] = ...) -> None: ...

class CopyIndexResponse(_message.Message):
    __slots__ = ("index",)
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class SortByField(_message.Message):
    __slots__ = ("field", "order")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    field: str
    order: _utils_pb2.Order
    def __init__(self, field: _Optional[str] = ..., order: _Optional[_Union[_utils_pb2.Order, str]] = ...) -> None: ...

class CreateFileEngineRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CreateMemoryEngineRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MappedField(_message.Message):
    __slots__ = ("source_field", "target_field")
    SOURCE_FIELD_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_FIELD_NUMBER: _ClassVar[int]
    source_field: str
    target_field: str
    def __init__(self, source_field: _Optional[str] = ..., target_field: _Optional[str] = ...) -> None: ...

class IndexAttributes(_message.Message):
    __slots__ = ("created_at", "unique_fields", "multi_fields", "description", "conflict_strategy", "mapped_fields", "auto_id_field")
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_FIELDS_FIELD_NUMBER: _ClassVar[int]
    MULTI_FIELDS_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    MAPPED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    AUTO_ID_FIELD_FIELD_NUMBER: _ClassVar[int]
    created_at: int
    unique_fields: _containers.RepeatedScalarFieldContainer[str]
    multi_fields: _containers.RepeatedScalarFieldContainer[str]
    description: str
    conflict_strategy: ConflictStrategy
    mapped_fields: _containers.RepeatedCompositeFieldContainer[MappedField]
    auto_id_field: str
    def __init__(self, created_at: _Optional[int] = ..., unique_fields: _Optional[_Iterable[str]] = ..., multi_fields: _Optional[_Iterable[str]] = ..., description: _Optional[str] = ..., conflict_strategy: _Optional[_Union[ConflictStrategy, str]] = ..., mapped_fields: _Optional[_Iterable[_Union[MappedField, _Mapping]]] = ..., auto_id_field: _Optional[str] = ...) -> None: ...

class CreateIndexRequest(_message.Message):
    __slots__ = ("index_name", "file", "memory", "schema", "compression", "blocksize", "sort_by_field", "index_attributes", "merge_policy", "query_parser_config")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    BLOCKSIZE_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_FIELD_NUMBER: _ClassVar[int]
    INDEX_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    MERGE_POLICY_FIELD_NUMBER: _ClassVar[int]
    QUERY_PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    file: CreateFileEngineRequest
    memory: CreateMemoryEngineRequest
    schema: str
    compression: Compression
    blocksize: int
    sort_by_field: SortByField
    index_attributes: IndexAttributes
    merge_policy: MergePolicy
    query_parser_config: _query_pb2.QueryParserConfig
    def __init__(self, index_name: _Optional[str] = ..., file: _Optional[_Union[CreateFileEngineRequest, _Mapping]] = ..., memory: _Optional[_Union[CreateMemoryEngineRequest, _Mapping]] = ..., schema: _Optional[str] = ..., compression: _Optional[_Union[Compression, str]] = ..., blocksize: _Optional[int] = ..., sort_by_field: _Optional[_Union[SortByField, _Mapping]] = ..., index_attributes: _Optional[_Union[IndexAttributes, _Mapping]] = ..., merge_policy: _Optional[_Union[MergePolicy, _Mapping]] = ..., query_parser_config: _Optional[_Union[_query_pb2.QueryParserConfig, _Mapping]] = ...) -> None: ...

class CreateIndexResponse(_message.Message):
    __slots__ = ("index",)
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class DeleteDocumentsRequest(_message.Message):
    __slots__ = ("index_name", "query")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    query: _query_pb2.Query
    def __init__(self, index_name: _Optional[str] = ..., query: _Optional[_Union[_query_pb2.Query, _Mapping]] = ...) -> None: ...

class DeleteDocumentsResponse(_message.Message):
    __slots__ = ("deleted_documents",)
    DELETED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    deleted_documents: int
    def __init__(self, deleted_documents: _Optional[int] = ...) -> None: ...

class DeleteIndexRequest(_message.Message):
    __slots__ = ("index_name",)
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class DeleteIndexResponse(_message.Message):
    __slots__ = ("deleted_index_name",)
    DELETED_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    deleted_index_name: str
    def __init__(self, deleted_index_name: _Optional[str] = ...) -> None: ...

class GetIndicesAliasesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetIndicesAliasesResponse(_message.Message):
    __slots__ = ("indices_aliases",)
    class IndicesAliasesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INDICES_ALIASES_FIELD_NUMBER: _ClassVar[int]
    indices_aliases: _containers.ScalarMap[str, str]
    def __init__(self, indices_aliases: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GetIndexRequest(_message.Message):
    __slots__ = ("index_name",)
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class GetIndexResponse(_message.Message):
    __slots__ = ("index",)
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: IndexDescription
    def __init__(self, index: _Optional[_Union[IndexDescription, _Mapping]] = ...) -> None: ...

class GetIndicesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetIndicesResponse(_message.Message):
    __slots__ = ("index_names",)
    INDEX_NAMES_FIELD_NUMBER: _ClassVar[int]
    index_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, index_names: _Optional[_Iterable[str]] = ...) -> None: ...

class IndexDocumentStreamRequest(_message.Message):
    __slots__ = ("index_name", "documents", "conflict_strategy", "skip_updated_at_modification")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_STRATEGY_FIELD_NUMBER: _ClassVar[int]
    SKIP_UPDATED_AT_MODIFICATION_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    documents: _containers.RepeatedScalarFieldContainer[bytes]
    conflict_strategy: ConflictStrategy
    skip_updated_at_modification: bool
    def __init__(self, index_name: _Optional[str] = ..., documents: _Optional[_Iterable[bytes]] = ..., conflict_strategy: _Optional[_Union[ConflictStrategy, str]] = ..., skip_updated_at_modification: bool = ...) -> None: ...

class IndexDocumentStreamResponse(_message.Message):
    __slots__ = ("elapsed_secs", "success_docs", "failed_docs")
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_DOCS_FIELD_NUMBER: _ClassVar[int]
    FAILED_DOCS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    success_docs: int
    failed_docs: int
    def __init__(self, elapsed_secs: _Optional[float] = ..., success_docs: _Optional[int] = ..., failed_docs: _Optional[int] = ...) -> None: ...

class IndexDocumentRequest(_message.Message):
    __slots__ = ("index_name", "document", "skip_updated_at_modification")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    SKIP_UPDATED_AT_MODIFICATION_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    document: bytes
    skip_updated_at_modification: bool
    def __init__(self, index_name: _Optional[str] = ..., document: _Optional[bytes] = ..., skip_updated_at_modification: bool = ...) -> None: ...

class IndexDocumentResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MergeSegmentsRequest(_message.Message):
    __slots__ = ("index_name", "segment_ids")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    segment_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, index_name: _Optional[str] = ..., segment_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class MergeSegmentsResponse(_message.Message):
    __slots__ = ("segment_id",)
    SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
    segment_id: str
    def __init__(self, segment_id: _Optional[str] = ...) -> None: ...

class SetIndexAliasRequest(_message.Message):
    __slots__ = ("index_alias", "index_name")
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    index_name: str
    def __init__(self, index_alias: _Optional[str] = ..., index_name: _Optional[str] = ...) -> None: ...

class SetIndexAliasResponse(_message.Message):
    __slots__ = ("old_index_name",)
    OLD_INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    old_index_name: str
    def __init__(self, old_index_name: _Optional[str] = ...) -> None: ...

class DocumentsRequest(_message.Message):
    __slots__ = ("index_name", "fields", "query_filter")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    QUERY_FILTER_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    fields: _containers.RepeatedScalarFieldContainer[str]
    query_filter: _query_pb2.Query
    def __init__(self, index_name: _Optional[str] = ..., fields: _Optional[_Iterable[str]] = ..., query_filter: _Optional[_Union[_query_pb2.Query, _Mapping]] = ...) -> None: ...

class DocumentsResponse(_message.Message):
    __slots__ = ("document",)
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    document: str
    def __init__(self, document: _Optional[str] = ...) -> None: ...

class VacuumIndexRequest(_message.Message):
    __slots__ = ("index_name", "excluded_segments")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    EXCLUDED_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    excluded_segments: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, index_name: _Optional[str] = ..., excluded_segments: _Optional[_Iterable[str]] = ...) -> None: ...

class VacuumIndexResponse(_message.Message):
    __slots__ = ("freed_space_bytes",)
    FREED_SPACE_BYTES_FIELD_NUMBER: _ClassVar[int]
    freed_space_bytes: int
    def __init__(self, freed_space_bytes: _Optional[int] = ...) -> None: ...

class WarmupIndexRequest(_message.Message):
    __slots__ = ("index_name", "is_full")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_FULL_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    is_full: bool
    def __init__(self, index_name: _Optional[str] = ..., is_full: bool = ...) -> None: ...

class WarmupIndexResponse(_message.Message):
    __slots__ = ("elapsed_secs",)
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: float
    def __init__(self, elapsed_secs: _Optional[float] = ...) -> None: ...

class FileEngineConfig(_message.Message):
    __slots__ = ("path",)
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class MemoryEngineConfig(_message.Message):
    __slots__ = ("schema",)
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    schema: str
    def __init__(self, schema: _Optional[str] = ...) -> None: ...

class CacheConfig(_message.Message):
    __slots__ = ("cache_size",)
    CACHE_SIZE_FIELD_NUMBER: _ClassVar[int]
    cache_size: int
    def __init__(self, cache_size: _Optional[int] = ...) -> None: ...

class RemoteEngineConfig(_message.Message):
    __slots__ = ("method", "url_template", "headers_template", "cache_config", "timeout_ms")
    class HeadersTemplateEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    METHOD_FIELD_NUMBER: _ClassVar[int]
    URL_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    HEADERS_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    CACHE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    method: str
    url_template: str
    headers_template: _containers.ScalarMap[str, str]
    cache_config: CacheConfig
    timeout_ms: int
    def __init__(self, method: _Optional[str] = ..., url_template: _Optional[str] = ..., headers_template: _Optional[_Mapping[str, str]] = ..., cache_config: _Optional[_Union[CacheConfig, _Mapping]] = ..., timeout_ms: _Optional[int] = ...) -> None: ...

class LogMergePolicy(_message.Message):
    __slots__ = ("is_frozen",)
    IS_FROZEN_FIELD_NUMBER: _ClassVar[int]
    is_frozen: bool
    def __init__(self, is_frozen: bool = ...) -> None: ...

class TemporalMergePolicy(_message.Message):
    __slots__ = ("merge_older_then_secs",)
    MERGE_OLDER_THEN_SECS_FIELD_NUMBER: _ClassVar[int]
    merge_older_then_secs: int
    def __init__(self, merge_older_then_secs: _Optional[int] = ...) -> None: ...

class IndexEngineConfig(_message.Message):
    __slots__ = ("file", "memory", "remote", "merge_policy", "query_parser_config")
    FILE_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    REMOTE_FIELD_NUMBER: _ClassVar[int]
    MERGE_POLICY_FIELD_NUMBER: _ClassVar[int]
    QUERY_PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    file: FileEngineConfig
    memory: MemoryEngineConfig
    remote: RemoteEngineConfig
    merge_policy: MergePolicy
    query_parser_config: _query_pb2.QueryParserConfig
    def __init__(self, file: _Optional[_Union[FileEngineConfig, _Mapping]] = ..., memory: _Optional[_Union[MemoryEngineConfig, _Mapping]] = ..., remote: _Optional[_Union[RemoteEngineConfig, _Mapping]] = ..., merge_policy: _Optional[_Union[MergePolicy, _Mapping]] = ..., query_parser_config: _Optional[_Union[_query_pb2.QueryParserConfig, _Mapping]] = ...) -> None: ...

class IndexDescription(_message.Message):
    __slots__ = ("index_name", "index_aliases", "index_engine", "num_docs", "compression", "index_attributes")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIASES_FIELD_NUMBER: _ClassVar[int]
    INDEX_ENGINE_FIELD_NUMBER: _ClassVar[int]
    NUM_DOCS_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    index_aliases: _containers.RepeatedScalarFieldContainer[str]
    index_engine: IndexEngineConfig
    num_docs: int
    compression: Compression
    index_attributes: IndexAttributes
    def __init__(self, index_name: _Optional[str] = ..., index_aliases: _Optional[_Iterable[str]] = ..., index_engine: _Optional[_Union[IndexEngineConfig, _Mapping]] = ..., num_docs: _Optional[int] = ..., compression: _Optional[_Union[Compression, str]] = ..., index_attributes: _Optional[_Union[IndexAttributes, _Mapping]] = ...) -> None: ...

class IndexDocumentOperation(_message.Message):
    __slots__ = ("document",)
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    document: bytes
    def __init__(self, document: _Optional[bytes] = ...) -> None: ...

class IndexOperation(_message.Message):
    __slots__ = ("index_document",)
    INDEX_DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    index_document: IndexDocumentOperation
    def __init__(self, index_document: _Optional[_Union[IndexDocumentOperation, _Mapping]] = ...) -> None: ...
