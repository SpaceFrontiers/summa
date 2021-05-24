from izihawa_nlptools.language_detect import detect_language
from izihawa_nlptools.utils import despace_full

from aiosumma.context import QueryContext
from aiosumma.parser import default_parser


class ProcessedQuery:
    def __init__(self, structured_query, context):
        self.structured_query = structured_query
        self.context = context

    def is_empty(self):
        return not bool(self.structured_query)

    def to_summa_query(self):
        return self.structured_query.to_summa_query() if self.structured_query else {'all': {}}


class QueryProcessor:
    def __init__(self, transformers=None):
        self.transformers = transformers or []

    def process(self, query, language):
        query = despace_full(query)
        structured_query = default_parser.parse(query.lower()) if query else None
        context = QueryContext(language=detect_language(query) or language)
        for transformer in self.transformers:
            structured_query = transformer.visit(structured_query, context=context)
        return ProcessedQuery(structured_query=structured_query, context=context)
