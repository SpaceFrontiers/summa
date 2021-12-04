from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

import utils_pb2 as _utils_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

Async: CommitMode
Brotli: Compression
DESCRIPTOR: _descriptor.FileDescriptor
File: IndexEngine
Lz4: Compression
Memory: IndexEngine
None: Compression
Snappy: Compression
Sync: CommitMode
Zstd: Compression

class AlterIndexRequest(_message.Message):
    __slots__ = ["blocksize", "compression", "default_fields", "index_name", "multi_fields", "primary_key", "sort_by_field"]
    class Fields(_message.Message):
        __slots__ = ["fields"]
        FIELDS_FIELD_NUMBER: _ClassVar[int]
        fields: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, fields: _Optional[_Iterable[str]] = ...) -> None: ...
    BLOCKSIZE_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    MULTI_FIELDS_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_FIELD_NUMBER: _ClassVar[int]
    blocksize: int
    compression: Compression
    default_fields: AlterIndexRequest.Fields
    index_name: str
    multi_fields: AlterIndexRequest.Fields
    primary_key: str
    sort_by_field: SortByField
    def __init__(self, index_name: _Optional[str] = ..., compression: _Optional[_Union[Compression, str]] = ..., blocksize: _Optional[int] = ..., sort_by_field: _Optional[_Union[SortByField, _Mapping]] = ..., default_fields: _Optional[_Union[AlterIndexRequest.Fields, _Mapping]] = ..., multi_fields: _Optional[_Union[AlterIndexRequest.Fields, _Mapping]] = ..., primary_key: _Optional[str] = ...) -> None: ...

class AlterIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: Index
    def __init__(self, index: _Optional[_Union[Index, _Mapping]] = ...) -> None: ...

class AttachIndexRequest(_message.Message):
    __slots__ = ["index_name"]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class AttachIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: Index
    def __init__(self, index: _Optional[_Union[Index, _Mapping]] = ...) -> None: ...

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
    __slots__ = ["autocommit_interval_ms", "blocksize", "compression", "default_fields", "index_engine", "index_name", "multi_fields", "primary_key", "schema", "sort_by_field", "writer_heap_size_bytes", "writer_threads"]
    AUTOCOMMIT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    BLOCKSIZE_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    INDEX_ENGINE_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    MULTI_FIELDS_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_FIELD_FIELD_NUMBER: _ClassVar[int]
    WRITER_HEAP_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    WRITER_THREADS_FIELD_NUMBER: _ClassVar[int]
    autocommit_interval_ms: int
    blocksize: int
    compression: Compression
    default_fields: _containers.RepeatedScalarFieldContainer[str]
    index_engine: IndexEngine
    index_name: str
    multi_fields: _containers.RepeatedScalarFieldContainer[str]
    primary_key: str
    schema: str
    sort_by_field: SortByField
    writer_heap_size_bytes: int
    writer_threads: int
    def __init__(self, index_name: _Optional[str] = ..., schema: _Optional[str] = ..., index_engine: _Optional[_Union[IndexEngine, str]] = ..., primary_key: _Optional[str] = ..., default_fields: _Optional[_Iterable[str]] = ..., compression: _Optional[_Union[Compression, str]] = ..., blocksize: _Optional[int] = ..., writer_heap_size_bytes: _Optional[int] = ..., writer_threads: _Optional[int] = ..., autocommit_interval_ms: _Optional[int] = ..., sort_by_field: _Optional[_Union[SortByField, _Mapping]] = ..., multi_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: Index
    def __init__(self, index: _Optional[_Union[Index, _Mapping]] = ...) -> None: ...

class DeleteDocumentRequest(_message.Message):
    __slots__ = ["index_alias", "primary_key"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    primary_key: PrimaryKey
    def __init__(self, index_alias: _Optional[str] = ..., primary_key: _Optional[_Union[PrimaryKey, _Mapping]] = ...) -> None: ...

class DeleteDocumentResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeleteIndexRequest(_message.Message):
    __slots__ = ["index_name"]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    def __init__(self, index_name: _Optional[str] = ...) -> None: ...

class DeleteIndexResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetIndexRequest(_message.Message):
    __slots__ = ["index_alias"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    def __init__(self, index_alias: _Optional[str] = ...) -> None: ...

class GetIndexResponse(_message.Message):
    __slots__ = ["index"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    index: Index
    def __init__(self, index: _Optional[_Union[Index, _Mapping]] = ...) -> None: ...

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
    indices: _containers.RepeatedCompositeFieldContainer[Index]
    def __init__(self, indices: _Optional[_Iterable[_Union[Index, _Mapping]]] = ...) -> None: ...

class Index(_message.Message):
    __slots__ = ["compression", "index_aliases", "index_engine", "index_name", "num_docs"]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIASES_FIELD_NUMBER: _ClassVar[int]
    INDEX_ENGINE_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_DOCS_FIELD_NUMBER: _ClassVar[int]
    compression: Compression
    index_aliases: _containers.RepeatedScalarFieldContainer[str]
    index_engine: str
    index_name: str
    num_docs: int
    def __init__(self, index_name: _Optional[str] = ..., index_aliases: _Optional[_Iterable[str]] = ..., index_engine: _Optional[str] = ..., num_docs: _Optional[int] = ..., compression: _Optional[_Union[Compression, str]] = ...) -> None: ...

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

class IndexOperation(_message.Message):
    __slots__ = ["index_document"]
    INDEX_DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    index_document: IndexDocumentOperation
    def __init__(self, index_document: _Optional[_Union[IndexDocumentOperation, _Mapping]] = ...) -> None: ...

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

class PrimaryKey(_message.Message):
    __slots__ = ["i64", "str"]
    I64_FIELD_NUMBER: _ClassVar[int]
    STR_FIELD_NUMBER: _ClassVar[int]
    i64: int
    str: str
    def __init__(self, str: _Optional[str] = ..., i64: _Optional[int] = ...) -> None: ...

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

class Compression(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class IndexEngine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class CommitMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
