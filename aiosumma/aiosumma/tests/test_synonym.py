from aiosumma import QueryProcessor
from aiosumma.tree_transformers import (
    SynonymTreeTransformer
)


def test_synonyms():
    query_processor = QueryProcessor(
        tree_transformers=[SynonymTreeTransformer.drugs()],
    )
    processed_query = query_processor.process('aspirin test', 'en')
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{'occur': 'should',
                                    'query': {'boolean': {'subqueries': [{'occur': 'should',
                                                                          'query': {
                                                                              'match': {'value': '"acetylsalicylic '
                                                                                                 'acid"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {
                                                                              'match': {'value': '"acetylsalisylic '
                                                                                                 'acid"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin '
                                                                                                       'sodium"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin '
                                                                                                       'magnesium"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin '
                                                                                                       'potassium"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin '
                                                                                                       'calcium"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin '
                                                                                                       'anhydride"'}}},
                                                                         {'occur': 'should',
                                                                          'query': {'match': {'value': '"aspirin '
                                                                                                       'lysine"'}}}]}}},
                                   {'occur': 'should',
                                    'query': {'match': {'value': 'test'}}}]},
    }

    processed_query = query_processor.process('walking on the road', 'en')
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{
            'occur': 'should', 'query': {'match': {'value': 'walking'}}}, {
            'occur': 'should', 'query': {'match': {'value': 'on'}}}, {
            'occur': 'should', 'query': {'match': {'value': 'the'}}}, {
            'occur': 'should', 'query': {'match': {'value': 'road'}}}]},
    }
