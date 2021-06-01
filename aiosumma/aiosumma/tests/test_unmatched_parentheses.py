from aiosumma.text_transformers import UnmatchedParenthesesTextTransformer


def test_unmatched_parentheses():
    text_transformer = UnmatchedParenthesesTextTransformer()
    assert text_transformer.process('[[[') == ''
    assert text_transformer.process(']]]') == ''
    assert text_transformer.process('((term1') == 'term1'
    assert text_transformer.process('((term1)') == '(term1)'
    assert text_transformer.process('((term1)))(term2') == '((term1))term2'
    assert text_transformer.process('term1)))(term2') == 'term1term2'
    assert text_transformer.process('[[[term1)))(term2>') == 'term1term2'
