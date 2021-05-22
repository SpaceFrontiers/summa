from aiosumma.parser.elements import (
    Boost,
    Phrase,
    SearchField,
    Word,
)
from aiosumma.tree_visitor import TreeTransformer


class ExactMatchTransformer(TreeTransformer):
    def __init__(self, default_phrase_field=None, score=1.0, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.default_phrase_field = default_phrase_field
        self.score = score

    def visit_group(self, node, context, parents=None):
        words = []
        phrase = []
        if len(node.operands) <= 1:
            return node, False
        for operand in node.operands:
            if not isinstance(operand, Word):
                return node, False
            words.append(operand)
            phrase.append(operand.value)
        phrase = ' '.join(phrase)
        if self.default_phrase_field:
            words.append(Boost(SearchField(self.default_phrase_field, Phrase(phrase)), self.score))
        else:
            words.append(Boost(Phrase(phrase), self.score))
        node.operands = words
        return node, False
