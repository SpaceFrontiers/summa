from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PBLink(_message.Message):
    __slots__ = ("hash", "name", "t_size")
    HASH_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    T_SIZE_FIELD_NUMBER: _ClassVar[int]
    hash: bytes
    name: str
    t_size: int
    def __init__(self, hash: _Optional[bytes] = ..., name: _Optional[str] = ..., t_size: _Optional[int] = ...) -> None: ...

class PBNode(_message.Message):
    __slots__ = ("links", "data")
    LINKS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    links: _containers.RepeatedCompositeFieldContainer[PBLink]
    data: bytes
    def __init__(self, links: _Optional[_Iterable[_Union[PBLink, _Mapping]]] = ..., data: _Optional[bytes] = ...) -> None: ...
