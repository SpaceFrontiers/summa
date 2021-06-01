from textblob import Word as TextBlobWord
from ..parser.elements import Word
from .base import TreeTransformer


class SpellcheckTreeTransformer(TreeTransformer):
    def visit_group(self, node, context, parents=None):
        if context.language == 'en':
            for operand in node.operands:
                if isinstance(operand, Word):
                    operand.value = str(TextBlobWord(operand.value).correct())
        return node, False
