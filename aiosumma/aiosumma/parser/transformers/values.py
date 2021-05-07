from typing import List, Set, Tuple

from aiosumma.parser.tree_visitor import TreeTransformer


class ValuePredicateWordTransformer:
    def node_predicate(self, node):
        raise NotImplementedError()

    def transform(self, node, context, parents, predicate_result):
        raise NotImplementedError()

    def apply(self, node, context, parents):
        if predicate_result := self.node_predicate(node):
            return self.transform(node, context, parents, predicate_result)


class ValueWordWordTransformer(ValuePredicateWordTransformer):
    def __init__(self, node_value):
        if isinstance(node_value, (List, Set, Tuple)):
            node_value_set = set(node_value)
            self.predicate = lambda node: node.value in node_value_set
        else:
            self.predicate = lambda node: node.value == node_value

    def node_predicate(self, node):
        return self.predicate(node.value)


class ContextWordTransformer(ValueWordWordTransformer):
    def __init__(self, node_value, context_transform):
        super().__init__(node_value=node_value)
        self.context_transform = context_transform

    def transform(self, node, context, parents, predicate_result):
        self.context_transform(context)
        return node


class ValuesWordTransformer(TreeTransformer):
    def __init__(self, word_transformers, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.word_transformers = word_transformers

    def visit_word(self, node, context, parents=None):
        for word_transformer in self.word_transformers:
            word_transformer.apply(node, context, parents=parents)
