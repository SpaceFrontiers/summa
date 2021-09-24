from textblob import Word as TextBlobWord

from ..parser.elements import (
    Group,
    SynonymsGroup,
    Word,
)
from .base import TreeTransformer


class SpellcheckTreeTransformer(TreeTransformer):
    def visit_group(self, node, context, parents=None):
        if context.query_language == 'en':
            corrected_operands = []
            for operand in node.operands:
                if isinstance(operand, Word):
                    corrected_value = str(TextBlobWord(operand.value).correct())
                    if corrected_value != operand.value:
                        operand = SynonymsGroup(operand, Word(corrected_value))
                corrected_operands.append(operand)
            return Group(*corrected_operands), True
        return node, False
