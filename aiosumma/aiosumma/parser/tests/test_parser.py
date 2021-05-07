from aiosumma.parser import ClassicQueryProcessor
from aiosumma.parser.tree import (
    Boost,
    Group,
    Phrase,
    Range,
    SearchField,
    Word,
)


def classic_query_processor(query):
    return ClassicQueryProcessor().process(query, 'en')


def test_doi_query():
    assert str(classic_query_processor('10.1001/azeroth1021.azerty').structured_query) == 'doi:10.1001/azeroth1021.azerty'
    assert str(classic_query_processor('https://doi.org/10.1001/azeroth1021.azerty').structured_query) == (
        'doi:10.1001/azeroth1021.azerty'
    )
    assert str(classic_query_processor('Gimme https://doi.org/10.1001/azeroth1021.azerty please').structured_query) == (
        '(gimme doi:10.1001/azeroth1021.azerty please)'
    )
    assert str(classic_query_processor('Gimme https://doi.org/10.1001/azeroth1021.azerty 10.6666/kek please').structured_query) == (
        '(gimme doi:10.1001/azeroth1021.azerty doi:10.6666/kek please)'
    )
    assert str(classic_query_processor('10.1001 / test').structured_query) == 'doi:10.1001/test'
    assert str(classic_query_processor('kek  10.1001  /  test').structured_query) == '(kek 10.1001 test)'


def test_isbn_query():
    assert str(classic_query_processor('ISBN: 9784567890123').structured_query) == '(isbn isbns:9784567890123)'
    assert str(classic_query_processor('ISBN:9784567890123').structured_query) == 'isbns:9784567890123'
    assert str(classic_query_processor('ISBN: 978-4567890123').structured_query) == '(isbn isbns:9784567890123)'
    assert str(classic_query_processor('9784567890123').structured_query) == 'isbns:9784567890123'
    assert str(classic_query_processor('978-4567890123').structured_query) == 'isbns:9784567890123'


def test_year_query():
    assert str(classic_query_processor('hemoglobin 2011').structured_query) == '(hemoglobin 2011 year:2011)'
    assert str(classic_query_processor('hemoglobin year:[2011 TO *]').structured_query) == '(hemoglobin year:[2011 TO *])'


def test_url():
    assert str(classic_query_processor('https://www.google.com/lelkek').structured_query) == (
        'https://www.google.com/lelkek'
    )
    assert str(classic_query_processor('http://www.google.com/lelkek').structured_query) == (
        'http://www.google.com/lelkek'
    )
    assert str(classic_query_processor('www.google.com/lelkek').structured_query) == (
        'www.google.com/lelkek'
    )
    assert str(classic_query_processor('telegram://t.me/com').structured_query) == (
        '(telegram ://t.me/com)'
    )


def test_emoji():
    processed_science_query = classic_query_processor('🔬 science query')
    assert str(processed_science_query.structured_query) == '(science query)'
    assert processed_science_query.context.index_aliases == ['scimag']

    processed_book_query = classic_query_processor('📚 book query')
    assert str(processed_book_query.structured_query) == '(book query)'
    assert processed_book_query.context.index_aliases == ['scitech']

    processed_science_and_book_query = classic_query_processor('🔬 📚 science or book query')
    assert str(processed_science_and_book_query.structured_query) == '(science or book query)'
    assert processed_science_and_book_query.context.index_aliases == ['scimag', 'scitech']

    another_emoji_query = classic_query_processor('😂 query')
    assert str(another_emoji_query.structured_query) == '(😂 query)'
    assert another_emoji_query.context.index_aliases == []


def test_default():
    processed_query_1 = classic_query_processor('“the ultimate trait for science”')
    assert processed_query_1.structured_query == Phrase('the ultimate trait for science')
    assert processed_query_1.json() == {'match': {'value': '"the ultimate trait for science"'}}

    processed_query_2 = classic_query_processor('“the ultimate” trait for science')
    assert processed_query_2.structured_query == Group(Phrase('the ultimate'), Word('trait'), Word('for'), Word('science'))
    assert processed_query_2.json() == {'bool': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': '"the ultimate"'}}},
        {'occur': 'should', 'query': {'match': {'value': 'trait'}}},
        {'occur': 'should', 'query': {'match': {'value': 'for'}}},
        {'occur': 'should', 'query': {'match': {'value': 'science'}}},
    ]}}

    processed_query_3 = classic_query_processor('Search "the ultimate trait for science"')
    assert processed_query_3.structured_query == Group(Word('search'), Phrase('the ultimate trait for science'))
    assert processed_query_3.json() == {
        'bool': {'subqueries': [{'occur': 'should', 'query': {'match': {'value': 'search'}}},
                                {'occur': 'should', 'query': {'match': {'value': '"the ultimate trait for science"'}}}]}}

    processed_query_4 = classic_query_processor(
        'hemoglobin blood issued_at:978307200^0.65 '
        'issued_at:[1262304000 TO 1577836800]^0.65 '
        'wrong_field : 123 spaced_1: 123 spaced :2 title:"title name"'
    )
    assert str(processed_query_4.structured_query) == (
        '(hemoglobin blood issued_at:978307200^0.65 issued_at:[1262304000 TO 1577836800]^0.65 '
        'wrong_field 123 spaced_1 123 spaced 2 title:"title name")'
    )
    assert processed_query_4.structured_query == Group(
        Word('hemoglobin'),
        Word('blood'),
        Boost(SearchField('issued_at', Word('978307200')), 0.65),
        Boost(SearchField('issued_at', Range(Word('1262304000'), Word('1577836800'))), 0.65),
        Word('wrong_field'),
        Word('123'),
        Word('spaced_1'),
        Word('123'),
        Word('spaced'),
        Word('2'),
        SearchField('title', Phrase('title name'))
    )
    assert processed_query_4.json() == {'bool': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'hemoglobin'}}},
        {'occur': 'should', 'query': {'match': {'value': 'blood'}}},
        {'occur': 'should', 'query': {'boost': {
            'force': '0.65',
            'query': {'term': {'field': 'issued_at', 'value': '978307200'}}}}},
        {'occur': 'should', 'query': {'boost': {
            'force': '0.65',
            'query': {'range': {'issued_at': {'gte': Word('1262304000'), 'lte': Word('1577836800')}}}}}},
        {'occur': 'should', 'query': {'match': {'value': 'wrong_field'}}},
        {'occur': 'should', 'query': {'match': {'value': '123'}}},
        {'occur': 'should', 'query': {'match': {'value': 'spaced_1'}}},
        {'occur': 'should', 'query': {'match': {'value': '123'}}},
        {'occur': 'should', 'query': {'match': {'value': 'spaced'}}},
        {'occur': 'should', 'query': {'match': {'value': '2'}}},
        {'occur': 'should', 'query': {'phrase': {'field': 'title', 'value': 'title name'}}},
    ]}}

    processed_query_5 = classic_query_processor('Science: In The Space')
    assert str(processed_query_5.structured_query) == '(science in the space)'
    assert processed_query_5.structured_query == Group(Word('science'), Word('in'), Word('the'), Word('space'))

    processed_query_6 = classic_query_processor("Glass's Guide to Commercial Vehicles 1989")
    assert str(processed_query_6.structured_query) == '(glass s guide to commercial vehicles 1989 year:1989)'
    assert processed_query_6.structured_query == Group(
        Word('glass'), Word('s'), Word('guide'), Word('to'),
        Word('commercial'), Word('vehicles'),
        Word('1989'),
        SearchField('year', Word('1989'))
    )
    processed_query_7 = classic_query_processor('(term1 +term2) term3 (title:term4 +term5 -(term6 term7))')
    assert processed_query_7.json() == {
        'bool': {'subqueries': [
            {'occur': 'should', 'query': {'match': {'value': 'term1'}}},
            {'occur': 'must', 'query': {'match': {'value': 'term2'}}},
            {'occur': 'should', 'query': {'match': {'value': 'term3'}}},
            {'occur': 'should', 'query': {'term': {'field': 'title', 'value': 'term4'}}},
            {'occur': 'must', 'query': {'match': {'value': 'term5'}}},
            {'occur': 'must_not',
             'query': {'bool': {'subqueries': [{'occur': 'should',
                                                'query': {'match': {'value': 'term6'}}},
                                               {'occur': 'should',
                                                'query': {'match': {'value': 'term7'}}}]}}}]}}

    processed_query_8 = classic_query_processor('title:(+term1 term2)')
    assert str(processed_query_8.structured_query) == '(+title:term1 title:term2)'
