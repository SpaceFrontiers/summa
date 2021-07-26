from ..parser.elements import Group
from .base import TreeTransformer


class OptimizingTreeTransformer(TreeTransformer):
    """
    Removes redundant groups
    """
    def visit_group(self, node, context, parents=None):
        merged_new_group = True
        while merged_new_group:
            new_operands = []
            merged_new_group = False
            for operand in node.operands:
                if isinstance(operand, Group):
                    merged_new_group = True
                    new_operands.extend(operand.operands)
                else:
                    new_operands.append(operand)
            node.operands = new_operands
        return node, False

    def visit_boost(self, node, context, parents=None):
        if node.score == 1.0:
            return node.expr, False
        return node, False
