from .base import TextTransformer


class UnmatchedParenthesesTextTransformer(TextTransformer):
    counterparts = {
        ')': '(',
        ']': '[',
        '>': '<',
    }

    def process(self, text: str):
        parentheses_indices = {
            '(': [],
            '[': [],
            '<': [],
        }
        processed_query = []
        index_shift = 0
        for index, ch in enumerate(text):
            if ch in parentheses_indices:
                parentheses_indices[ch].append(index - index_shift)
                processed_query.append(ch)
            elif ch in self.counterparts:
                if len(parentheses_indices[self.counterparts[ch]]) > 0:
                    parentheses_indices[self.counterparts[ch]] = parentheses_indices[self.counterparts[ch]][:-1]
                    processed_query.append(ch)
                else:
                    index_shift += 1
            else:
                processed_query.append(ch)
        delete_indices = []
        for paren in parentheses_indices:
            delete_indices.extend(parentheses_indices[paren])
        if delete_indices:
            delete_indices = sorted(delete_indices, reverse=True)
        for ix in delete_indices:
            del processed_query[ix]
        return ''.join(processed_query)
