---
title: Query DSL
parent: Core
nav_order: 1
has_toc: true
---

Query describes what documents you want to extract from Summa for further processing. The stream of matched documents are then passed
to [collectors](/summa/cors/collectors) responsible for final processing. So you should learn the difference between queries and collectors,
first ones tell what documents should be taken from database and second ones describes how to process them and what to return to users.

There are different kinds of possible queries. 
One kind of queries are just scoring and returning documents. 
For example, `TermQuery` matches all documents containing specified term and associates score to every matched document. These scores
may be used for ranking documents by relevance.
Other queries such as `BooleanQuery` or `DisjunctionMaxQuery` combine documents and scores matched by multiple sub-queries.

## TermQuery
The most basic kind of query. 
Match documents that contain the specified term (word) inside the specified field.
Every matched document also associated with its BM25 score relevant to the query.

```json 
{
  "term": {
    "field": "title", 
    "value": "astronomy"
  }
}
```

## BooleanQuery
Allows to combine multiple queries into a single one. Every sub-query has a property named `occur` describing how to do combination

- `must` tells that all matched documents must match to this sub-query too
- `must_not` tells that all matched documents must not contain documents matching to this sub-query
- `should` tells that matched documents may contain documents matching to this sub-query

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
    }, {
      "occur": "must_not", 
      "query": {
        "phrase": {
          "field": "author",
          "value": "tony igy"
        }
      }
    }]
  }
}
```

## DisjunctionMaxQuery
Allows to combine multiple queries into a single one. It is similar to `BooleanQuery` but scores are calculated in other way.
Instead of summarizing scores of all sub-queries, it takes maximum score of a single sub-query. Such approach may be useful
in specific cases like searching documents with synonyms.

```json 
{
  "disjunction_max": {
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
          "value": "astronomia"
        }
      }
    }, {
      "occur": "must_not", 
      "query": {
        "phrase": {
          "field": "author",
          "value": "tony igy"
        }
      }
    }]
  }
}
```

## BoostQuery
Modifies scores produced by a nested query. Useful in `BooleanQuery` to penalize or boost
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
`MatchQuery` is a special query. Summa takes the value of this query, parses it and produces other kind of queries.
`MatchQuery` may be used for parsing queries written in natural language. For example, following query
```json
{
  "match": {
      "value": "astronomy +nebula -\"tony igy\""
  }
}
```
will be parsed into
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
    }, {
      "occur": "must_not", 
      "query": {
        "phrase": {
          "field": "author",
          "value": "tony igy"
        }
      }
    }]
  }
}
```

## PhraseQuery
Documents containing exact occurrence of the phrase
```json
{
  "phrase": {
    "field": "title",
    "value": "general astronomy"
  }
}
```

## RegexQuery
Documents that have field value matched against the regular expression
```json
{
  "regex": {
    "field": "category",
    "value": "book.*"
  }
}
```

## RangeQuery
Documents where the requested field lays between the range
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
Documents that look like passed document

```json
{
  "more_like_this": {
    "document": "{\"title\": \"astronomy\"}"
  }
}
```

## AllQuery
All documents

```json
{
  "all": {}
}
```
