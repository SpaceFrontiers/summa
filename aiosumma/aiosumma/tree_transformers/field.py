from ..parser.elements import (
    Group,
    Word,
)
from .base import TreeTransformer


class FieldTreeTransformer(TreeTransformer):
    def __init__(self, field_aliases, valid_fields, invalid_fields=None, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.field_aliases = field_aliases
        self.valid_fields = valid_fields
        self.invalid_fields = invalid_fields

    def visit_search_field(self, node, context, parents=None):
        if node.name in self.field_aliases:
            node.name = self.field_aliases[node.name]

        if node.name in self.valid_fields:
            return node, False

        if self.invalid_fields and node.name in self.invalid_fields:
            context.has_invalid_fields = True
            return None, False

        return Group(Word(node.name), node.expr), False
