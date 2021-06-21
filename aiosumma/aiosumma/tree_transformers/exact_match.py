from typing import Callable, Union

from ..parser.elements import (
    Boost,
    Phrase,
    SearchField,
    Word,
)
from .base import TreeTransformer


def _default_scorer(node, context) -> float:
    return float(len(node))


class ExactMatchTreeTransformer(TreeTransformer):
    def __init__(self, default_phrase_field=None, score: Union[float, Callable] = _default_scorer, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.default_phrase_field = default_phrase_field
        self.score = score

    def visit_group(self, node, context, parents=None):
        words = []
        phrase = []
        if len(node) <= 1:
            return node, False
        for operand in node.operands:
            if not isinstance(operand, Word):
                return node, False
            words.append(operand)
            phrase.append(operand.value)
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
