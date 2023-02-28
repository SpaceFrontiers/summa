from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Data(_message.Message):
    __slots__ = ("type", "data", "filesize", "blocksizes", "hashType", "fanout")
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Raw: _ClassVar[Data.DataType]
        Directory: _ClassVar[Data.DataType]
        File: _ClassVar[Data.DataType]
        Metadata: _ClassVar[Data.DataType]
        Symlink: _ClassVar[Data.DataType]
        HAMTShard: _ClassVar[Data.DataType]
    Raw: Data.DataType
    Directory: Data.DataType
    File: Data.DataType
    Metadata: Data.DataType
    Symlink: Data.DataType
    HAMTShard: Data.DataType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    FILESIZE_FIELD_NUMBER: _ClassVar[int]
    BLOCKSIZES_FIELD_NUMBER: _ClassVar[int]
    HASHTYPE_FIELD_NUMBER: _ClassVar[int]
    FANOUT_FIELD_NUMBER: _ClassVar[int]
    type: Data.DataType
    data: bytes
    filesize: int
    blocksizes: _containers.RepeatedScalarFieldContainer[int]
    hashType: int
    fanout: int
    def __init__(self, type: _Optional[_Union[Data.DataType, str]] = ..., data: _Optional[bytes] = ..., filesize: _Optional[int] = ..., blocksizes: _Optional[_Iterable[int]] = ..., hashType: _Optional[int] = ..., fanout: _Optional[int] = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ("MimeType",)
    MIMETYPE_FIELD_NUMBER: _ClassVar[int]
    MimeType: str
    def __init__(self, MimeType: _Optional[str] = ...) -> None: ...
