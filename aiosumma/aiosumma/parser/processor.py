from aiosumma.parser.context import QueryContext
from aiosumma.parser.parser import default_parser
from aiosumma.parser.tree import (
    Doi,
    Group,
    Url,
    Word,
)
from izihawa_nlptools.utils import despace_full


class ProcessedQuery:
    def __init__(self, structured_query, context):
        self.structured_query = structured_query
        self.context = context

    def json(self):
        return self.structured_query.json()

    def __repr__(self):
        return repr(self.structured_query)

    def __str__(self):
        return str(self.structured_query)

    def is_exploration_query(self):
        return isinstance(self.structured_query, (Group, Word)) and not isinstance(self.structured_query, (Url, Doi))


class QueryProcessor:
    def __init__(self, transformers=None):
        self.transformers = transformers or []

    def process(self, query, language):
        query = despace_full(query)
        structured_query = default_parser.parse(query.lower())
        context = QueryContext(language=language)
        for transformer in self.transformers:
            structured_query = transformer.visit(structured_query, context=context)
        return ProcessedQuery(structured_query=structured_query, context=context)
