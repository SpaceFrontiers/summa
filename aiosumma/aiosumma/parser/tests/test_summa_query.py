import pytest

from aiosumma.errors import UnsupportedQueryError
from aiosumma.parser import default_parser


def test_default():
    parsed_query = default_parser.parse('search engine')
    assert parsed_query.to_summa_query() == {
        'bool': {'subqueries': [
            {'occur': 'should', 'query': {'match': {'value': 'search'}}},
            {'occur': 'should', 'query': {'match': {'value': 'engine'}}}
        ]}}


def test_plus_minus():
    parsed_query = default_parser.parse('-search +engine')
    assert parsed_query.to_summa_query() == {'bool': {'subqueries': [
        {'occur': 'must_not', 'query': {'match': {'value': 'search'}}},
        {'occur': 'must', 'query': {'match': {'value': 'engine'}}}
    ]}}


def test_search_field():
    parsed_query = default_parser.parse('title:kolobok')
    assert parsed_query.to_summa_query() == {'term': {'field': 'title', 'value': 'kolobok'}}


def test_search_field_with_group():
    with pytest.raises(UnsupportedQueryError):
        default_parser.parse('title:(kolobok babushka)').to_summa_query()


def test_not_regex():
    parsed_query = default_parser.parse('pyth.*')
    assert parsed_query.to_summa_query() == {'match': {'value': 'pyth.*'}}


def test_regex():
    parsed_query = default_parser.parse('title:/pyth.*/')
    assert parsed_query.to_summa_query() == {'regex': {'field': 'title', 'value': 'pyth.*'}}


def test_free_regex():
    parsed_query = default_parser.parse('/pyth.*/')
    assert parsed_query.to_summa_query() == {'match': {'value': '/pyth.*/'}}


def test_free_with_slash_regex():
    parsed_query = default_parser.parse('/pyth/.*/')
    assert parsed_query.to_summa_query() == {'bool': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': '/pyth/'}}},
        {'occur': 'should', 'query': {'match': {'value': '.*/'}}}
    ]}}


def test_free_with_escaped_slash_regex():
    parsed_query = default_parser.parse('/pyth\\/.*/')
    assert parsed_query.to_summa_query() == {'match': {'value': '/pyth/.*/'}}


def test_phrase():
    parsed_query = default_parser.parse('title:"kolobok"')
    assert parsed_query.to_summa_query() == {'phrase': {'field': 'title', 'value': 'kolobok'}}


def test_free_phrase():
    parsed_query = default_parser.parse('"kolobok"')
    assert parsed_query.to_summa_query() == {'match': {'value': '"kolobok"'}}


def test_boost():
    parsed_query = default_parser.parse('title:kolobok^3.0')
    assert parsed_query.to_summa_query() == {'boost': {
        'query': {'term': {'field': 'title', 'value': 'kolobok'}},
        'score': '3.00000'
    }}


def test_boost_group():
    parsed_query = default_parser.parse('(kolobok babushka)^3.0')
    assert parsed_query.to_summa_query() == {'boost': {'query': {'bool': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'kolobok'}}},
        {'occur': 'should', 'query': {'match': {'value': 'babushka'}}}
    ]}}, 'score': '3.00000'}}


def test_proximity():
    parsed_query = default_parser.parse('"Kolobok babushka"~3')
    assert parsed_query.to_summa_query() == {'match': {'value': '"Kolobok babushka"~3'}}
