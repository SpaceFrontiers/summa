from aiosumma.parser.tree import (
    Group,
    Minus,
    Plus,
    SearchField,
    Word,
)
from aiosumma.parser.tree_visitor import TreeTransformer


class TantivyTransformer(TreeTransformer):
    def visit_phrase(self, node, context, parents=None):
        splitted = node.value.split()
        if len(splitted) == 0:
            return None
        if len(splitted) == 1:
            return Word(splitted[0])
        return node

    def visit_search_field(self, node, context, parents=None):
        if isinstance(node.expr, Group):
            return Group(*[
                SearchField(node.name, operand) for operand in node.expr.operands
            ])
        if isinstance(node.expr, Plus):
            return Plus(SearchField(node.name, node.expr.a))
        if isinstance(node.expr, Minus):
            return Minus(SearchField(node.name, node.expr.a))
        return node
