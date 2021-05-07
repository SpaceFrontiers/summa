from aiosumma.parser.tree import (
    Group,
    Word,
)
from aiosumma.parser.tree_visitor import TreeTransformer


class FieldTransformer(TreeTransformer):
    FIELD_ALIASES = {
        'author': 'authors',
        'isbn': 'isbns',
        'journal': 'container_title',
        'lang': 'language',
        'body': 'content',
        'refs': 'ref_by_count',
    }

    VALID_FIELDS = frozenset([
        'id', 'abstract', 'authors', 'container_title', 'content',
        'doi', 'description', 'isbns', 'issued_at', 'language', 'original_id',
        'ref_by_count', 'references', 'tags', 'title', 'year',
    ])

    def visit_search_field(self, node, context, parents=None):
        if node.name in self.FIELD_ALIASES:
            node.name = self.FIELD_ALIASES[node.name]

        if node.name not in self.VALID_FIELDS:
            return Group(Word(node.name), node.expr)

        return node
