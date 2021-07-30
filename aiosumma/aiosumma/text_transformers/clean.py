from izihawa_nlptools.utils import remove_hidden_chars

from .base import TextTransformer


class CleanTextTransformer(TextTransformer):
    def process(self, text: str):
        return remove_hidden_chars(text)


