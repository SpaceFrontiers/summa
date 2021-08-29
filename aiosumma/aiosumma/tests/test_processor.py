from aiosumma import QueryProcessor
from aiosumma.text_transformers import LowerTextTransformer
from aiosumma.tree_transformers import (
    DoiTreeTransformer,
    ExactMatchTreeTransformer,
    MorphyTreeTransformer,
    OptimizingTreeTransformer,
    OrderByTreeTransformer,
    SynonymTreeTransformer,
    TantivyTreeTransformer,
    ValuesWordTreeTransformer,
    ValueWordTreeTransformer,
)


class MarkWordTransformer(ValueWordTreeTransformer):
    def __init__(self):
        super().__init__(node_value='mark')

    def transform(self, node, context, parents, predicate_result):
        context.is_forced_clean = True
        return None


def test_optimizing_query_processor():
    query_processor = QueryProcessor()
    processed_query = query_processor.process('search engine', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'boolean': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'search'}}},
        {'occur': 'should', 'query': {'match': {'value': 'engine'}}}
    ]}}
    processed_query = query_processor.process('(search (dog cat))', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'boolean': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'search'}}},
        {'occur': 'should', 'query': {'match': {'value': 'dog'}}},
        {'occur': 'should', 'query': {'match': {'value': 'cat'}}},
    ]}}


def test_order_by_query_processor():
    query_processor = QueryProcessor(
        tree_transformers=[OrderByTreeTransformer(
            field_aliases={
                'f1': 'field1',
                'f2': 'field2',
            },
            valid_fields=frozenset(['field1', 'field2', 'field3']),
        )],
    )
    processed_query = query_processor.process('term1 term2 order_by:f1', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'boolean': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'term1'}}},
        {'occur': 'should', 'query': {'match': {'value': 'term2'}}}
    ]}}
    assert processed_query.context.order_by == ('field1', 'desc')


def test_values_processor():
    query_processor = QueryProcessor(tree_transformers=[
        ValuesWordTreeTransformer(word_transformers=[MarkWordTransformer()]),
    ])
    processed_query = query_processor.process('term1 term2 mark', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'boolean': {'subqueries': [
        {'occur': 'should', 'query': {'match': {'value': 'term1'}}},
        {'occur': 'should', 'query': {'match': {'value': 'term2'}}}
    ]}}
    assert processed_query.context.is_forced_clean


def test_production_chain():
    query_processor = QueryProcessor(
        text_transformers=[LowerTextTransformer()],
        tree_transformers=[
            ExactMatchTreeTransformer('title'),
            MorphyTreeTransformer(),
            SynonymTreeTransformer.drugs(),
            TantivyTreeTransformer(),
            OptimizingTreeTransformer(),
        ],
    )
    processed_query = query_processor.process('Claudio rugarli', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{'occur': 'should',
                                    'query': {'match': {'value': 'claudio'}}},
                                   {'occur': 'should',
                                    'query': {'match': {'value': 'rugarli'}}},
                                   {'occur': 'should',
                                    'query': {'boost': {'query': {'phrase': {'field': 'title',
                                                                             'slop': 1,
                                                                             'value': 'claudio '
                                                                                      'rugarli'}},
                                                        'score': '2'}}}]},
    }

    processed_query = query_processor.process('+(search engine) -car', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{'occur': 'must',
                                    'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'search'}},
                                                                                {'match': {'value': 'searches'}}]}}},
                                   {'occur': 'must',
                                    'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'engine'}},
                                                                                {'match': {'value': 'engines'}}]}}},
                                   {'occur': 'must',
                                    'query': {'boost': {'query': {'phrase': {'field': 'title',
                                                                             'slop': 1,
                                                                             'value': 'search '
                                                                                      'engine'}},
                                                        'score': '2'}}},
                                   {'occur': 'should',
                                    'query': {'disjunction_max': {
                                        'disjuncts': [{'boolean': {'subqueries': [{'occur': 'must_not',
                                                                                   'query': {
                                                                                       'match': {'value': 'car'}}}]}},
                                                      {'boolean': {'subqueries': [{'occur': 'must_not',
                                                                                   'query': {'match': {
                                                                                       'value': 'cars'}}}]}}]}}}]},
    }
    processed_query = query_processor.process(
        '(Editor), (Editor), (Editor)',
        languages={'en': 1.0},
    )
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{
            'occur': 'should',
            'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'editor'}},
                                                        {'match': {'value': 'editors'}}]}}},
            {'occur': 'should',
             'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'editor'}},
                                                         {'match': {'value': 'editors'}}]}}},
            {'occur': 'should',
             'query': {'disjunction_max': {'disjuncts': [{'match': {'value': 'editor'}},
                                                         {'match': {'value': 'editors'}}]}}}]},
    }


def test_unknown_language_transformer():
    query_processor = QueryProcessor(
        tree_transformers=[MorphyTreeTransformer(enable_morph=True), OptimizingTreeTransformer()])
    processed_query = query_processor.process('search engine', languages={'zz': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [
            {'occur': 'should', 'query': {'disjunction_max': {'disjuncts': [
                {'match': {'value': 'search'}},
                {'match': {'value': 'searches'}}]}}},
            {'occur': 'should', 'query': {'disjunction_max': {'disjuncts': [
                {'match': {'value': 'engine'}},
                {'match': {'value': 'engines'}}]}}},
        ]}
    }


def test_unknown_query_language_transformer():
    query_processor = QueryProcessor(
        tree_transformers=[MorphyTreeTransformer(enable_morph=True), OptimizingTreeTransformer()])
    processed_query = query_processor.process('kavanaba mutagor', languages={'zz': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [
            {'occur': 'should', 'query': {'match': {'value': 'kavanaba'}}},
            {'occur': 'should', 'query': {'match': {'value': 'mutagor'}}}]},
    }


def test_exact_match_transformers():
    query_processor = QueryProcessor(
        tree_transformers=[
            ExactMatchTreeTransformer('title'),
        ]
    )
    processed_query = query_processor.process('search engine', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {
        'boolean': {'subqueries': [{'occur': 'should',
                                    'query': {'match': {'value': 'search'}}},
                                   {'occur': 'should',
                                    'query': {'match': {'value': 'engine'}}},
                                   {'occur': 'should',
                                    'query': {'boost': {'query': {'phrase': {'field': 'title',
                                                                             'slop': 1,
                                                                             'value': 'search '
                                                                                      'engine'}},
                                                        'score': '2'}}}]},
    }


def test_doi_transformer():
    query_processor = QueryProcessor(
        tree_transformers=[
            DoiTreeTransformer(),
        ]
    )
    processed_query = query_processor.process('https://doi.org/10.1101/2022.05.26.493559', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'boost': {'query': {'term': {
        'field': 'doi',
        'value': '10.1101/2022.05.26.493559',
    }}, 'score': '1'}}
    processed_query = query_processor.process('https://google.com/?query=one+two+three', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'match': {'value': '"https://google.com/?query=one+two+three"'}}
    processed_query = query_processor.process('https://doi.org/10.1101', languages={'en': 1.0})
    assert processed_query.to_summa_query() == {'match': {'value': '"https://doi.org/10.1101"'}}
