from izihawa_nlptools.language_detect import detect_language

from aiosumma.context import QueryContext
from aiosumma.errors import ParserError
from aiosumma.parser import default_parser
from aiosumma.parser.errors import ParseError


class ProcessedQuery:
    def __init__(self, structured_query, context):
        self.structured_query = structured_query
        self.context = context

    def is_empty(self):
        return not bool(self.structured_query)

    def to_summa_query(self):
        return self.structured_query.to_summa_query() if self.structured_query else {'all': {}}


class QueryProcessor:
    def __init__(self, text_transformers=None, tree_transformers=None):
        self.text_transformers = text_transformers or []
        self.tree_transformers = tree_transformers or []

    def process(self, query, language):
        if query is not None:
            for text_transformer in self.text_transformers:
                query = text_transformer.process(query)
            try:
                structured_query = default_parser.parse(query) if query else None
            except ParseError as error:
                raise ParserError(query=query, nested_error=error)
            context = QueryContext(language=detect_language(query) or language)
            for tree_transformer in self.tree_transformers:
                structured_query = tree_transformer.visit(structured_query, context=context)
            return ProcessedQuery(structured_query=structured_query, context=context)
        else:
            return ProcessedQuery(structured_query=None, context=QueryContext(language=language))
