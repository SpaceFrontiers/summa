import query_pb2 as _query_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ["index_alias", "query", "collectors", "is_fieldnorms_scoring_enabled", "load_cache", "store_cache"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    COLLECTORS_FIELD_NUMBER: _ClassVar[int]
    IS_FIELDNORMS_SCORING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    LOAD_CACHE_FIELD_NUMBER: _ClassVar[int]
    STORE_CACHE_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    query: _query_pb2.Query
    collectors: _containers.RepeatedCompositeFieldContainer[_query_pb2.Collector]
    is_fieldnorms_scoring_enabled: bool
    load_cache: bool
    store_cache: bool
    def __init__(self, index_alias: _Optional[str] = ..., query: _Optional[_Union[_query_pb2.Query, _Mapping]] = ..., collectors: _Optional[_Iterable[_Union[_query_pb2.Collector, _Mapping]]] = ..., is_fieldnorms_scoring_enabled: bool = ..., load_cache: bool = ..., store_cache: bool = ...) -> None: ...
