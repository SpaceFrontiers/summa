from typing import (
    Callable,
    Union,
)

from ..parser.elements import (
    Boost,
    Group,
    Phrase,
    Proximity,
    SearchField,
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
        phrase = []

        if len(node) <= 1:
            return node, False

        for operand in node.operands:
            if isinstance(operand, Word):
                phrase.append(operand.value)
            else:
                return node, False

        if not phrase:
            return node, False

        phrase = ' '.join(phrase)

        score = self.score
        if callable(score):
            score = score(node, context)

        if self.default_phrase_field:
            exact_query = Boost(SearchField(self.default_phrase_field, Proximity(Phrase(phrase), slop=1)), score)
        else:
            exact_query = Boost(Proximity(Phrase(phrase), slop=1), score)

        return Group(*node.operands, exact_query), False
