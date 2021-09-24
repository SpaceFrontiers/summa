from aiosumma import QueryProcessor
from aiosumma.tree_transformers import SpellcheckTreeTransformer


def test_spell_check():
    query_processor = QueryProcessor(
        tree_transformers=[SpellcheckTreeTransformer()],
    )
    processed_query = query_processor.process('desinged fro embeded usin', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{'occur': 'should',
                                    'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'desinged'}},
                                                                                {'match': {'value': 'designed'}}]}}},
                                   {'occur': 'should',
                                    'query': {'match': {'value': 'fro'}}},
                                   {'occur': 'should',
                                    'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'embeded'}},
                                                                                {'match': {'value': 'embedded'}}]}}},
                                   {'occur': 'should',
                                    'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'usin'}},
                                                                                {'match': {'value': 'using'}}]}}}]},
    }
    processed_query = query_processor.process('covid-19 haemophilia', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'boolean': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'covid-19'}}},
        {'occur': 'should', 'query': {'match': {'value': 'haemophilia'}}}]}
    }
