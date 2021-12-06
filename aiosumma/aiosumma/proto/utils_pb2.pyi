from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

Asc: Order
DESCRIPTOR: _descriptor.FileDescriptor
Desc: Order

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class Order(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
