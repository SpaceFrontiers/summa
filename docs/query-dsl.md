---
layout: page
subtitle: Query DSL
permalink: /query-dsl
---

There are different kinds of possible queries Summa can process. 
One kind of queries are just scoring and returning documents like `TermQuery` 
that returns all documents with some term scored with BM25. Other queries can
combine nested queries and modify their scores like `BoostQuery` or `BooleanQuery`

## TermQuery

The most basic kind of query. 
`TermQuery` scores and returns documents that contain the specific word inside specific field.
The list of documents are ranged according to BM25 score.
```json 
{"term": {
    "field": "title", 
    "value": "astronomy"
}}
```

## BooleanQuery

`BooleanQuery` allowes to combine multiple queries into a single one.
```json 
{"boolean": {"subqueries": [
    {"occur": "should", "query": {
        "term": {
            "field": "title",
            "value": "astronomy"
        }
    }},
    {"occur": "must", "query": {
        "term": {
            "field": "title",
            "value": "nebula"
        }
    }}
]}}
```

## BoostQuery

`BoostQuery` modifies scores produced by a nested query

## MatchQuery

`MatchQuery` uses Tantivy parser to create tree of other queries

## PhraseQuery

`PhraseQuery` matches documents containing exact occurrence of the phrase

## RegexQuery

## RangeQuery

## MoreLikeThisQuery