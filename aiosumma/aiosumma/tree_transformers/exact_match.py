from typing import Callable, Union

from ..parser.elements import (
    Boost,
    Phrase,
    SearchField,
    SynonymsGroup,
    Word,
)
from .base import TreeTransformer


def _default_scorer(node, context) -> str:
    return str(len(node))


class ExactMatchTreeTransformer(TreeTransformer):
    def __init__(self, default_phrase_field=None, score: Union[str, Callable] = _default_scorer, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.default_phrase_field = default_phrase_field
        self.score = score

    def visit_group(self, node, context, parents=None):
        words = []
        phrase = []
        if len(node) <= 1:
            return node, False
        for operand in node.operands:
            if isinstance(operand, Word):
                words.append(operand)
                phrase.append(operand.value)
            elif isinstance(operand, SynonymsGroup):
                words.append(operand.operands[0])
                phrase.append(operand.operands[0].value)
            else:
                return node, False
        phrase = ' '.join(phrase)

        score = self.score
        if callable(score):
            score = score(node, context)
        if self.default_phrase_field:
            words.append(Boost(SearchField(self.default_phrase_field, Phrase(phrase)), score))
        else:
            words.append(Boost(Phrase(phrase), score))
        node.operands = words
        return node, False
