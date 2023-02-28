from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetTopTermsRequest(_message.Message):
    __slots__ = ("index_name", "field_name", "top_k")
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    index_name: str
    field_name: str
    top_k: int
    def __init__(self, index_name: _Optional[str] = ..., field_name: _Optional[str] = ..., top_k: _Optional[int] = ...) -> None: ...

class GetTopTermsResponse(_message.Message):
    __slots__ = ("per_segment",)
    class PerSegmentEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: SegmentTerms
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[SegmentTerms, _Mapping]] = ...) -> None: ...
    PER_SEGMENT_FIELD_NUMBER: _ClassVar[int]
    per_segment: _containers.MessageMap[str, SegmentTerms]
    def __init__(self, per_segment: _Optional[_Mapping[str, SegmentTerms]] = ...) -> None: ...

class SegmentTerms(_message.Message):
    __slots__ = ("term_infos",)
    TERM_INFOS_FIELD_NUMBER: _ClassVar[int]
    term_infos: _containers.RepeatedCompositeFieldContainer[TermInfo]
    def __init__(self, term_infos: _Optional[_Iterable[_Union[TermInfo, _Mapping]]] = ...) -> None: ...

class TermInfo(_message.Message):
    __slots__ = ("key", "doc_freq")
    KEY_FIELD_NUMBER: _ClassVar[int]
    DOC_FREQ_FIELD_NUMBER: _ClassVar[int]
    key: bytes
    doc_freq: int
    def __init__(self, key: _Optional[bytes] = ..., doc_freq: _Optional[int] = ...) -> None: ...
