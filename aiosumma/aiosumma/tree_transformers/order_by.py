from ..parser.elements import Word
from .base import TreeTransformer


class OrderByTreeTransformer(TreeTransformer):
    """
    Set order_by in `QueryContext` and removes corresponding node from the query
    """
    def __init__(self, field_aliases, valid_fields, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.field_aliases = field_aliases
        self.valid_fields = valid_fields

    def visit_search_field(self, node, context, parents=None):
        if node.name == 'order_by':
            if isinstance(node.expr, Word):
                if node.expr.value in self.field_aliases:
                    node.expr.value = self.field_aliases[node.expr.value]
                if node.expr.value in self.valid_fields:
                    context.order_by = (node.expr.value, 'desc')
                    return None, True
        return node, False
