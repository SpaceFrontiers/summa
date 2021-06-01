from izihawa_nlptools.utils import despace_full

from .base import TextTransformer


class DespaceTextTransformer(TextTransformer):
    def process(self, text: str):
        return despace_full(text)
