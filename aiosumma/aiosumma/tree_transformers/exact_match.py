from typing import (
    Callable,
    Union,
)

from ..parser.elements import (
    Boost,
    Phrase,
    Proximity,
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
        new_operands = []
        phrase = []
        phrase_len = len(node)
        if phrase_len <= 1:
            return node, False
        for operand in node.operands:
            new_operands.append(operand)
            if isinstance(operand, Word):
                phrase.append(operand.value)
            elif isinstance(operand, SynonymsGroup):
                phrase.append(operand.operands[0].value)
            elif isinstance(operand, SearchField):
                continue
            else:
                return node, False

        if not phrase:
            return node, False

        phrase = ' '.join(phrase)

        score = self.score
        if callable(score):
            score = score(node, context)
        if self.default_phrase_field:
            new_operands.append(Boost(SearchField(self.default_phrase_field, Proximity(Phrase(phrase), slop=1)), score))
        else:
            new_operands.append(Boost(Proximity(Phrase(phrase), slop=1), score))
        node.operands = new_operands
        return node, False
