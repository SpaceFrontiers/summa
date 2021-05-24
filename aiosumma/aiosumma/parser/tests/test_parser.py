import pytest

from aiosumma.parser import default_parser
from aiosumma.parser.elements import (
    Boost,
    Doi,
    Group,
    Minus,
    Phrase,
    Plus,
    Proximity,
    Regex,
    SearchField,
    Url,
    Word,
)
from aiosumma.parser.errors import ParseError


def test_default():
    parsed_query = default_parser.parse('search engine')
    assert parsed_query == Group(Word('search'), Word('engine'))


def test_plus_minus():
    parsed_query = default_parser.parse('-search +engine')
    assert parsed_query == Group(Minus(Word('search')), Plus(Word('engine')))


def test_nested_group_1():
    parsed_query = default_parser.parse('search (cat dog)')
    assert parsed_query == Group(Word('search'), Word('cat'), Word('dog'))


def test_nested_group_2():
    parsed_query = default_parser.parse('(search (cat mouse (dog rhino)))')
    assert parsed_query == Group(Word('search'), Word('cat'), Word('mouse'), Word('dog'), Word('rhino'))


def test_search_field():
    parsed_query = default_parser.parse('title:kolobok')
    assert parsed_query == SearchField('title', Word('kolobok'))


def test_search_field_with_group():
    parsed_query = default_parser.parse('title:(kolobok babushka)')
    assert parsed_query == SearchField('title', Group(Word('kolobok'), Word('babushka')))


def test_group_as_search_field():
    with pytest.raises(ParseError):
        default_parser.parse('(title body):ninja')


def test_plus_minus_group():
    parsed_query = default_parser.parse('+(term1 term2) -(term3 term4)')
    assert parsed_query == Group(Plus(Group(Word('term1'), Word('term2'))), Minus(Group(Word('term3'), Word('term4'))))


def test_large_plus_minus_group():
    parsed_query = default_parser.parse('+(term1 term2 term3 term4 term5) -(term6 term7 term8 term9 term10)')
    assert parsed_query == Group(
        Plus(Group(Word('term1'), Word('term2'), Word('term3'), Word('term4'), Word('term5'))),
        Minus(Group(Word('term6'), Word('term7'), Word('term8'), Word('term9'), Word('term10'))),
    )


def test_not_regex():
    parsed_query = default_parser.parse('pyth.*')
    assert parsed_query == Word('pyth.*')


def test_regex():
    parsed_query = default_parser.parse('title:/pyth.*/')
    assert parsed_query == SearchField('title', Regex('pyth.*'))


def test_free_regex():
    parsed_query = default_parser.parse('/pyth.*/')
    assert parsed_query == Regex('pyth.*')


def test_free_with_slash_regex():
    parsed_query = default_parser.parse('/pyth/.*/')
    assert parsed_query == Group(Regex('pyth'), Word('.*/'))


def test_free_with_escaped_slash_regex():
    parsed_query = default_parser.parse('/pyth\\/.*/')
    assert parsed_query == Regex('pyth/.*')


def test_phrase():
    parsed_query = default_parser.parse('title:"kolobok"')
    assert parsed_query == SearchField('title', Phrase('kolobok'))


def test_free_phrase():
    parsed_query = default_parser.parse('"kolobok"')
    assert parsed_query == Phrase("kolobok")


def test_boost():
    parsed_query = default_parser.parse('title:kolobok^3.0')
    assert parsed_query == Boost(SearchField('title', Word('kolobok')), 3)


def test_boost_group():
    parsed_query = default_parser.parse('(kolobok babushka)^3.0')
    assert parsed_query == Boost(Group(Word('kolobok'), Word('babushka')), 3)


def test_proximity():
    parsed_query = default_parser.parse('"Kolobok babushka"~3')
    assert parsed_query == Proximity(Phrase('Kolobok babushka'), 3)


def test_doi():
    parsed_query = default_parser.parse('10.1385/nmm:9:1:17')
    assert parsed_query == Doi('10.1385/nmm:9:1:17')


def test_url():
    parsed_query = default_parser.parse('https://doi.org/10.1101/2022.05.26.493559')
    assert parsed_query == Url('https://doi.org/10.1101/2022.05.26.493559')
