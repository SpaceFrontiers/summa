from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class Order(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Asc: _ClassVar[Order]
    Desc: _ClassVar[Order]
Asc: Order
Desc: Order

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
