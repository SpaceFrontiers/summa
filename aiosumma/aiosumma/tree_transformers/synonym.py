import csv
import os.path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from ..parser.elements import (
    Group,
    Phrase,
    SynonymsGroup,
)
from .base import TreeTransformer


class SynonymTreeTransformer(TreeTransformer):
    def __init__(self, mapping: Dict[str, List[str]], ignore_nodes: Optional[Tuple] = None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.mapping = mapping

    @staticmethod
    def from_synlists(synlists: Iterable[List], ignore_nodes: Optional[Tuple] = None):
        mapping = {}
        for synlist in synlists:
            for term in synlist:
                if term in mapping:
                    raise ValueError(
                        'Synsets {current} and {previous} are overlapping'.format(
                            current=synlist,
                            previous=mapping[term],
                        )
                    )
                mapping[term] = synlist
        return SynonymTreeTransformer(mapping=mapping, ignore_nodes=ignore_nodes)

    @staticmethod
    def from_synlists_file(filepath: str, ignore_nodes: Optional[Tuple] = None):
        synlists = []
        with open(filepath, newline='') as csvfile:
            for synlist in csv.reader(csvfile):
                synlists.append(synlist)
        return SynonymTreeTransformer.from_synlists(synlists=synlists, ignore_nodes=ignore_nodes)

    @staticmethod
    def drugs(ignore_nodes: Optional[Tuple] = None):
        filepath = os.path.join(os.path.dirname(__file__), '../', 'data/synsets/drugs.csv')
        return SynonymTreeTransformer.from_synlists_file(filepath=filepath, ignore_nodes=ignore_nodes)

    def synonyms(self, term):
        return self.mapping.get(term)

    def visit_word(self, node, context, parents=None):
        if not parents or isinstance(parents[-1], Group):
            synset_list = self.synonyms(node.value)
            if synset_list is not None:
                words = list(map(lambda x: Phrase(x), synset_list))
                return SynonymsGroup(*words), True
        return node, False
