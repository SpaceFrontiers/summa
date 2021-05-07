from aiosumma.parser.tree import (
    Group,
    Word,
)
from aiosumma.parser.tree_visitor import TreeTransformer
from izihawa_nlptools.morph import EnglishMorphology


class MorphyTransformer(TreeTransformer):
    morphology = {
        'en': EnglishMorphology('en_core_web_sm'),
    }

    def __init__(self, is_morph=False, is_accent=True):
        super().__init__()
        self.is_morph = is_morph
        self.is_accent = is_accent

    def visit_word(self, node, context, parents=None):
        if context.language not in self.morphology:
            return node
        if self.is_morph:
            forms = [Word(w, final=True) for w in self.morphology[context.language].derive_forms(node.value)]
            return Group(*forms)
        if self.is_accent:
            if 'ё' in node.value:
                forms = [Word(node.value, final=True), Word(node.value.replace('ё', 'е'), final=True)]
                return Group(*forms)
        return node
