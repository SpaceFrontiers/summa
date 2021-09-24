from aiosumma import QueryProcessor
from aiosumma.tree_transformers import SynonymTreeTransformer


def test_synonyms():
    query_processor = QueryProcessor(
        tree_transformers=[SynonymTreeTransformer.drugs()],
    )
    processed_query = query_processor.process('aspirin test', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{'occur': 'should',
                                    'query': {'disjunction_max': {'disjuncts': [{'match': {'value': '"acetylsalicylic '
                                                                                                    'acid"'}},
                                                                                {'match': {'value': '"acetylsalisylic '
                                                                                                    'acid"'}},
                                                                                {'match': {'value': '"aspirin"'}},
                                                                                {'match': {'value': '"aspirin '
                                                                                                    'sodium"'}},
                                                                                {'match': {'value': '"aspirin '
                                                                                                    'magnesium"'}},
                                                                                {'match': {'value': '"aspirin '
                                                                                                    'potassium"'}},
                                                                                {'match': {'value': '"aspirin '
                                                                                                    'calcium"'}},
                                                                                {'match': {'value': '"aspirin '
                                                                                                    'anhydride"'}},
                                                                                {'match': {'value': '"aspirin '
                                                                                                    'lysine"'}}]}}},
                                   {'occur': 'should',
                                    'query': {'match': {'value': 'test'}}}]},
    }

    processed_query = query_processor.process('walking on the road', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{
            'occur': 'should', 'query': {'match': {'value': 'walking'}}}, {
            'occur': 'should', 'query': {'match': {'value': 'on'}}}, {
            'occur': 'should', 'query': {'match': {'value': 'the'}}}, {
            'occur': 'should', 'query': {'match': {'value': 'road'}}}]},
    }
