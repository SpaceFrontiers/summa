from typing import (
    List,
    Set,
    Tuple,
)

from .base import TreeTransformer


class ValuePredicateWordTreeTransformer:
    def node_predicate(self, node):
        raise NotImplementedError()

    def transform(self, node, context, parents, predicate_result):
        raise NotImplementedError()


class ValueWordTreeTransformer(ValuePredicateWordTreeTransformer):
    def __init__(self, node_value):
        if isinstance(node_value, (List, Set, Tuple)):
            node_value_set = set(node_value)
            self.predicate = lambda node: node.value in node_value_set
        else:
            self.predicate = lambda node: node.value == node_value

    def node_predicate(self, node):
        return self.predicate(node)


class ContextWordTreeTransformer(ValueWordTreeTransformer):
    def __init__(self, node_value, context_transform):
        super().__init__(node_value=node_value)
        self.context_transform = context_transform

    def transform(self, node, context, parents, predicate_result):
        self.context_transform(context)
        return None


class ValuesWordTreeTransformer(TreeTransformer):
    def __init__(self, word_transformers, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.word_transformers = word_transformers

    def visit_word(self, node, context, parents=None):
        for word_transformer in self.word_transformers:
            if predicate_result := word_transformer.node_predicate(node):
                return word_transformer.transform(node, context, parents, predicate_result), True
        return node, False
