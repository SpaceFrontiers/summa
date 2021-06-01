from .base import TextTransformer


class LowerTextTransformer(TextTransformer):
    def process(self, text: str):
        return text.lower()
