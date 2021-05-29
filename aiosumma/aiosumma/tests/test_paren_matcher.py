from aiosumma.paren_matcher import remove_unmatched_parens


def test_paren_matcher():
    assert remove_unmatched_parens('[[[') == ''
    assert remove_unmatched_parens(']]]') == ''
    assert remove_unmatched_parens('((term1') == 'term1'
    assert remove_unmatched_parens('((term1)') == '(term1)'
    assert remove_unmatched_parens('((term1)))(term2') == '((term1))term2'
    assert remove_unmatched_parens('term1)))(term2') == 'term1term2'
    assert remove_unmatched_parens('[[[term1)))(term2>') == 'term1term2'
