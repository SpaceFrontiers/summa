from izihawa_nlptools.morph import EnglishMorphology

from aiosumma.parser.elements import (
    Boost,
    Group,
    Word,
)
from aiosumma.tree_visitor import TreeTransformer


class MorphyTransformer(TreeTransformer):
    """
    Creates forms of words
    """
    morphology = {
        'en': EnglishMorphology('en_core_web_sm'),
    }

    def __init__(self, enable_morph=True, enable_accent=True, ignore_nodes=None):
        super().__init__(ignore_nodes=ignore_nodes)
        self.enable_morph = enable_morph
        self.enable_accent = enable_accent

    def visit_word(self, node, context, parents=None):
        forms = [node]

        if self.enable_accent and 'ё' in node.value:
            forms.append(Word(node.value.replace('ё', 'е')))

        if self.enable_morph and context.language in self.morphology:
            for w in self.morphology[context.language].derive_forms(node.value):
                if node.value != w:
                    forms.append(Boost(Word(w), score=0.85))

        if len(forms) == 1:
            return node, True
        return Group(*forms), True
