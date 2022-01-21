---
title: Query DSL
parent: Core
nav_order: 2
has_toc: true
---

Query describes what documents you want to extract from Summa for further processing.
The stream of matched documents are then passed to [collectors](/summa/cors/collectors) responsible for final processing.
Therefore, it is important to understand the difference between queries and collectors.
Queries tell which documents to take from the database and collectors describe how to process them and what to return to the users.

There are different types of possible queries.
One type of query simply scores and returns documents.
For example, TermQuery matches all documents containing the specified term and associates a score with every matched document. These scores can be used to rank documents by relevance.
Other queries, such as BooleanQuery or DisjunctionMaxQuery, combine documents and scores matched by multiple sub-queries.

## TermQuery
The most basic kind of query is a `TermQuery`. 
This type of query matches all documents that contain the specified term (word) within the specified field. 
Every matched document is also associated with a BM25 score that is relevant to the query.
```json 
{
  "term": {
    "field": "title", 
    "value": "astronomy"
  }
}
```

## BooleanQuery
Allows combining multiple queries into a single one. Every sub-query has a property named `occur` that describes how to combine them.

- `must` tells that all matched documents must match this sub-query as well.
- `must_not` tells that all matched documents must not contain documents that match this sub-query.
- `should` tells that matched documents may contain documents that match this sub-query.
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
Modifies scores produced by a nested query. Useful in `BooleanQuery` to penalize or boost parts of the query.
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
