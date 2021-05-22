from aiosumma.parser.elements import (
    Boost,
    Group,
    Minus,
    Plus,
    SearchField,
    Word,
)
from aiosumma.tree_visitor import TreeTransformer


class TantivyTransformer(TreeTransformer):
    def visit_phrase(self, node, context, parents=None):
        splitted = node.value.split()
        if len(splitted) == 0:
            return None, False
        if len(splitted) == 1:
            return Word(splitted[0]), False
        context.is_exploration = False
        return node, False

    def visit_minus(self, node, context, parents=None):
        if isinstance(node.a, Group):
            op = node.a
            new_operands = []
            for operand in op.operands:
                new_operands.append(Minus(operand))
            op.operands = new_operands
            return op, False
        return node, False

    def visit_plus(self, node, context, parents=None):
        if isinstance(node.a, Group):
            op = node.a
            new_operands = []
            for operand in op.operands:
                new_operands.append(Plus(operand))
            op.operands = new_operands
            return op, False
        return node, False

    def visit_search_field(self, node, context, parents=None):
        context.is_exploration = False
        if isinstance(node.expr, Group):
            return Group(*[
                SearchField(node.name, operand) for operand in node.expr.operands
            ]), False
        if isinstance(node.expr, Plus):
            return Plus(SearchField(node.name, node.expr.a)), False
        if isinstance(node.expr, Minus):
            return Minus(SearchField(node.name, node.expr.a)), False
        if isinstance(node.expr, Boost):
            return Boost(SearchField(node.name, node.expr.expr), score=node.expr.score), False
        return node, False
