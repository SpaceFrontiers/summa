from typing import (
    List,
    Optional,
)

from izihawa_nlptools.language_detect import detect_language

from aiosumma.context import QueryContext
from aiosumma.parser import default_parser
from aiosumma.parser.errors import ParseError


class ProcessedQuery:
    """
    `ProcessedQuery` keeps together parsed query and its context
    """
    def __init__(self, parsed_query, context):
        self.parsed_query = parsed_query
        self.context = context

    def is_empty(self):
        # Check if there are any nodes in the query
        return not bool(self.parsed_query)

    def to_summa_query(self):
        return self.parsed_query.to_summa_query() if self.parsed_query else {'all': {}}

    def __repr__(self):
        return repr(self.parsed_query)


class QueryProcessor:
    """
    `QueryProcessor` processes the query written in the natural language to the structured query
    """
    def __init__(self, text_transformers: Optional[List] = None, tree_transformers: Optional[List] = None):
        """
        Args:
            text_transformers: list of text transformers that process text query before parsing
            tree_transformers: list of tree transformers that process syntactic tree after parsing
        """
        self.text_transformers = text_transformers or []
        self.tree_transformers = tree_transformers or []

    def parse(self, query: str):
        try:
            return default_parser.parse(query)
        except ParseError as error:
            error.add('query', query)
            error.add('nested_error', error)
            raise error

    def apply_text_transformers(self, query: str):
        for text_transformer in self.text_transformers:
            query = text_transformer.process(query)
        return query

    def apply_tree_transformers(self, parsed_query, context: QueryContext):
        for tree_transformer in self.tree_transformers:
            parsed_query = tree_transformer.visit(parsed_query, context=context)
        return parsed_query

    def process(self, query: str, language: Optional[str] = None) -> ProcessedQuery:
        if not query:
            return ProcessedQuery(parsed_query=None, context=QueryContext(language=language))

        query = self.apply_text_transformers(query=query)
        parsed_query = self.parse(query=query)
        context = QueryContext(language=detect_language(query, threshold=0.5) or language)
        parsed_query = self.apply_tree_transformers(parsed_query=parsed_query, context=context)

        return ProcessedQuery(parsed_query=parsed_query, context=context)
