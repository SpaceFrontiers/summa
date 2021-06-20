from aiosumma import QueryProcessor
from aiosumma.tree_transformers import SpellcheckTreeTransformer


def test_spell_check():
    query_processor = QueryProcessor(
        tree_transformers=[SpellcheckTreeTransformer()],
    )
    processed_query = query_processor.process('desinged fro embeded usin', 'en')
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{
            'occur': 'should',
            'query': {'boolean': {'subqueries': [{
                'occur': 'should',
                'query': {'match': {'value': 'designed'}}}, {
                'occur': 'should',
                'query': {'match': {'value': 'desinged'}}}]}}}, {
            'occur': 'should',
            'query': {'match': {'value': 'fro'}}}, {
            'occur': 'should',
            'query': {'boolean': {'subqueries': [{
                'occur': 'should',
                'query': {'match': {'value': 'embedded'}}}, {
                'occur': 'should',
                'query': {'match': {'value': 'embeded'}}}]}}}, {
            'occur': 'should',
            'query': {'boolean': {'subqueries': [{
                'occur': 'should',
                'query': {'match': {'value': 'using'}}}, {
                'occur': 'should',
                'query': {'match': {'value': 'usin'}}}]}}}]},
    }
    processed_query = query_processor.process('covid-19 haemophilia', 'en')
    assert processed_query.to_summa_query() == {'boolean': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'covid-19'}}},
        {'occur': 'should', 'query': {'match': {'value': 'haemophilia'}}}]}
    }
