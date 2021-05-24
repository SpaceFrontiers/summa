import re

from izihawa_nlptools.regex import (
    DOI_REGEX,
    DOI_WILDCARD_REGEX_TEXT,
)

from aiosumma.parser.elements import (
    Doi,
    Group,
    Minus,
    Plus,
    Regex,
    SearchField,
)
from aiosumma.tree_visitor import TreeTransformer

from .values import ValuePredicateWordTransformer


class DoiTransformer(TreeTransformer):
    def visit_doi(self, node, context, parents):
        if parents is None or len(parents) == 0 or isinstance(parents[-1], Group):
            context.dois.append(node.value)
            context.is_exploration = False
            return SearchField('doi', node), True
        return node, False

    def visit_url(self, node, context, parents):
        if match := re.search(DOI_REGEX, node.value):
            if parents is None or len(parents) == 0 or isinstance(parents[-1], Group):
                doi = (match[1] + '/' + match[2]).lower()
                context.dois.append(doi)
                context.is_exploration = False
                return SearchField('doi', Doi(doi)), True
        return node, False


class DoiWildcardWordTransformer(ValuePredicateWordTransformer):
    def node_predicate(self, node):
        return re.search(DOI_WILDCARD_REGEX_TEXT, node.value)

    def transform(self, node, context, parents, predicate_result):
        doi_prefix = predicate_result[0].lower()
        if parents is None or len(parents) == 0 or isinstance(parents[-1], Group):
            context.is_exploration = False
            return Plus(SearchField('doi', Regex(doi_prefix)))
        if isinstance(parents[-1], (Plus, Minus)):
            context.is_exploration = False
            return SearchField('doi', Regex(doi_prefix))
        return node
