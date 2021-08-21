from typing import (
    Dict,
    List,
    Optional,
    Union,
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
        return self.parsed_query.to_summa_query(self.context) if not self.is_empty() else self.context.blank_query()

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

    def apply_tree_transformers(self, processed_query: ProcessedQuery):
        for tree_transformer in self.tree_transformers:
            processed_query.parsed_query = tree_transformer.visit(processed_query.parsed_query, context=processed_query.context)
        return processed_query

    def process(self, query: str, languages: Optional[Union[Dict[str, float], str]] = None) -> ProcessedQuery:
        if languages is None or languages == '':
            languages = {}

        if isinstance(languages, str) and len(languages) > 0:
            languages = {languages: 1.0}

        if not query:
            return ProcessedQuery(parsed_query=None, context=QueryContext(languages=languages))

        query = self.apply_text_transformers(query=query)
        parsed_query = self.parse(query=query)
        if query_language := detect_language(query, threshold=0.5):
            languages = {**languages, query_language: 1.0}
        context = QueryContext(languages=languages, query_language=query_language)
        processed_query = ProcessedQuery(parsed_query=parsed_query, context=context)
        processed_query = self.apply_tree_transformers(processed_query)

        return processed_query
