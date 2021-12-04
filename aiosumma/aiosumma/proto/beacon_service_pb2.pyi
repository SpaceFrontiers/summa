from typing import ClassVar as _ClassVar
from typing import Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class PublishIndexRequest(_message.Message):
    __slots__ = ["index_alias", "payload"]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    index_alias: str
    payload: str
    def __init__(self, index_alias: _Optional[str] = ..., payload: _Optional[str] = ...) -> None: ...

class PublishIndexResponse(_message.Message):
    __slots__ = ["hash"]
    HASH_FIELD_NUMBER: _ClassVar[int]
    hash: str
    def __init__(self, hash: _Optional[str] = ...) -> None: ...
