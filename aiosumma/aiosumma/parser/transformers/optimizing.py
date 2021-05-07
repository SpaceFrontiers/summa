from aiosumma.parser.tree import Group
from aiosumma.parser.tree_visitor import TreeTransformer


class OptimizingTransformer(TreeTransformer):
    def collapse_group(self, node):
        new_operands = []
        for operand in node.operands:
            if isinstance(operand, Group):
                new_operands.extend(self.collapse_group(operand))
            else:
                new_operands.append(operand)
        return new_operands

    def visit_group(self, node, context, parents=None):
        if len(node.operands) == 0:
            return None
        elif len(node.operands) == 1:
            return node.operands[0]
        node.operands = tuple(self.collapse_group(node))
        return node
