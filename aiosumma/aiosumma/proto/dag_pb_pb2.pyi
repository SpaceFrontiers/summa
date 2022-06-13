from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class PBLink(_message.Message):
    __slots__ = ["hash", "name", "t_size"]
    HASH_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    T_SIZE_FIELD_NUMBER: _ClassVar[int]
    hash: bytes
    name: str
    t_size: int
    def __init__(self, hash: _Optional[bytes] = ..., name: _Optional[str] = ..., t_size: _Optional[int] = ...) -> None: ...

class PBNode(_message.Message):
    __slots__ = ["data", "links"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    links: _containers.RepeatedCompositeFieldContainer[PBLink]
    def __init__(self, links: _Optional[_Iterable[_Union[PBLink, _Mapping]]] = ..., data: _Optional[bytes] = ...) -> None: ...
