from ..parser.elements import (
    Group,
    Word,
)
from .base import TreeTransformer


class FieldTreeTransformer(TreeTransformer):
    def __init__(self, field_aliases, valid_fields, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.field_aliases = field_aliases
        self.valid_fields = valid_fields

    def visit_search_field(self, node, context, parents=None):
        if node.name in self.field_aliases:
            node.name = self.field_aliases[node.name]

        if node.name in self.valid_fields:
            return node, False

        return Group(Word(node.name), node.expr), False
