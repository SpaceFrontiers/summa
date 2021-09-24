import re
from typing import Optional

from izihawa_nlptools.regex import (
    DOI_REGEX,
    DOI_WILDCARD_REGEX_TEXT,
)

from ..parser.elements import (
    Boost,
    Doi,
    Group,
    Minus,
    Phrase,
    Plus,
    Regex,
    SearchField,
)
from .base import TreeTransformer
from .values import ValuePredicateWordTreeTransformer


class DoiTreeTransformer(TreeTransformer):
    def __init__(self, score: str = '1.0', ignore_nodes: Optional[tuple] = None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.score = score

    def visit_doi(self, node, context, parents):
        if parents is None or len(parents) == 0 or isinstance(parents[-1], Group):
            context.dois.append(node.value)
            return Boost(SearchField('doi', node), score=self.score), True
        return node, False

    def visit_url(self, node, context, parents):
        if match := re.search(DOI_REGEX, node.value):
            if parents is None or len(parents) == 0 or isinstance(parents[-1], Group):
                doi = (match[1] + '/' + match[2]).lower()
                context.dois.append(doi)
                return Boost(SearchField('doi', Doi(doi)), score=self.score), True
        return Phrase(node.value), False


class DoiWildcardWordTreeTransformer(ValuePredicateWordTreeTransformer):
    def node_predicate(self, node):
        return re.search(DOI_WILDCARD_REGEX_TEXT, node.value)

    def transform(self, node, context, parents, predicate_result):
        doi_prefix = predicate_result[0].lower()
        if parents is None or len(parents) == 0 or isinstance(parents[-1], Group):
            return Plus(SearchField('doi', Regex(doi_prefix)))
        if isinstance(parents[-1], (Plus, Minus)):
            return SearchField('doi', Regex(doi_prefix))
        return node
