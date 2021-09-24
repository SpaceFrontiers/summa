---
layout: page
title: Collectors
permalink: /collectors
---
Collectors are responsible for processing the stream of documents that matched the query. 
Every collector ingests this stream and derives output based on what ingested.
Simple collectors such as `Count` just counting documents in the stream, other collectors such as `TopDocs` maintaining a bounded set of documents havinng highest scores to return them to user.
The main goal of every collector is reducing the amount of data that should be returned to the requestor.
Though collectors looks through the whole set of matched documents, they are limiting the output in various ways.

## TopDocs
### Default Scoring
Plain BM25 with limit anf offset (can be omitted)
```json
{
  "top_docs": {
    "limit": 10, 
    "offset": 200
  }
}
```

### Order By
Top documents order by `FastField`
```json
{
  "top_docs": {
    "limit": 10,
    "scorer": {
      "order_by": "popularity_score"
    }
  }
}
```

### Eval Expression
Top documents order by `EvalExpr`
```json
{
  "top_docs": {
    "limit": 10, 
    "scorer": {
      "eval_expr": "original_score * log(e(), 1 + popularity_score)"
    }
  }
}
```

## Facets
Facet search on facet field

```json
{
  "facet": {
    "field": "category",
    "facets": ["/genre/fiction", "/topic/biology"]
  }
}
```


## Count
Counts the number of documents exactly and returns it
```json 
{"count": {}}
```

## Reservoir
Select `limit` random items corresponding to the query and returns them
```json
{"reservoir": {"limit": 10}}
```

## Aggregation
Returns an aggregation
```json
{"aggregation": {"aggregations": {"year_stats": {"metric": {"stats": {"field": "issued_at"}}}}}}
```