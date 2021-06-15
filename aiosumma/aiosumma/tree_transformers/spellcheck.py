from textblob import Word as TextBlobWord
from ..parser.elements import Group, Word
from .base import TreeTransformer


class SpellcheckTreeTransformer(TreeTransformer):
    def visit_group(self, node, context, parents=None):
        if context.language == 'en':
            corrected_operands = []
            for operand in node.operands:
                if isinstance(operand, Word):
                    corrected_value = str(TextBlobWord(operand.value).correct())
                    if corrected_value != operand.value:
                        operand = Group(Word(corrected_value), Word(operand.value))
                corrected_operands.append(operand)
            return Group(*corrected_operands), True
        return node, False
