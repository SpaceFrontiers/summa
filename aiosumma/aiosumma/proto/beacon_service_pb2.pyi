from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PublishIndexRequest(_message.Message):
    __slots__ = ["copy", "index_alias", "payload"]
    COPY_FIELD_NUMBER: _ClassVar[int]
    INDEX_ALIAS_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    copy: bool
    index_alias: str
    payload: str
    def __init__(self, index_alias: _Optional[str] = ..., copy: bool = ..., payload: _Optional[str] = ...) -> None: ...

class PublishIndexResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
