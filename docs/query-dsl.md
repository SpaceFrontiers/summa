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
{
  "term": {
    "field": "title", 
    "value": "astronomy"
  }
}
```

## BooleanQuery
`BooleanQuery` allowes to combine multiple queries into a single one.
```json 
{
  "boolean": {
    "subqueries": [{
      "occur": "should",
      "query": {
        "term": {
          "field": "title",
          "value": "astronomy"
        }
      }
    }, {
      "occur": "must", 
      "query": {
        "term": {
          "field": "title",
          "value": "nebula"
        }
      }
    }]
  }
}
```

## BoostQuery
`BoostQuery` modifies scores produced by a nested query. It is useful in `BooleanQuery` to penalize or boost
parts of the query.
```json
{
  "boolean": {
    "subqueries": [{
      "occur": "should",
      "query": {
       "boost": {
         "query": {
           "term": {
             "field": "title",
             "value": "astronomy"
           }
         },
         "score": "2.0"
       }
      }
    }, {
      "occur": "must", 
      "query": {
        "term": {
          "field": "title",
          "value": "nebula"
        }
      }
    }]
  }
}
```

## MatchQuery
`MatchQuery` uses Tantivy parser to create tree of other queries. 
```json
{
  "match": {
      "value": "astronomy"
  }
}
```

## PhraseQuery
`PhraseQuery` matches documents containing exact occurrence of the phrase
```json
{
  "phrase": {
    "field": "title",
    "value": "general astronomy"
  }
}
```

## RegexQuery
`RegexQuery` returns documents that has field value matched against the regular expression
```json
{
  "regex": {
    "field": "category",
    "value": "book.*"
  }
}
```

## RangeQuery
`RangeQuery` returns documents where the requested field lays between the range

```json
{
  "range": {
    "field": "create_timestamp",
    "range": {
      "left": "2021-01-01",
      "right": "2022-01-01"
    }  
  }
}
```

## MoreLikeThisQuery
`MoreLikeThisQuery` returns documents that looks like passed document

```json
{
  "more_like_this": {
    "document": "{\"title\": \"astronomy\"}"
  }
}
```