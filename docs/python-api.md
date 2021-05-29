---
layout: page
subtitle: Schema
permalink: /python-api
---

# Aiosumma
Aiosumma consists of two parts: Python client for GRPC API and query library. While Python client for GRPC API
is pretty plain, query library provides tools for extending queries and deriving intents through various tools

## Query Library
```python
import asyncio

from aiosumma import QueryProcessor, SummaClient
from aiosumma.transformers import MorphyTransformer
from izihawa_nlptools.language_detect import detect_language

async def main():
    # Create client and connect to the endpoint
    client = SummaClient('localhost:8082')
    await client.start_and_wait()
    
    # `QueryProcessor` converts textual queries into the tree that can be
    # further casted to Summa DSL Query
    # Transformers is an extra fuctionality attached to the `QueryProcessor` 
    # and responsible for particular tree mutations. 
    # For example, `MorphyTransformer` can convert `Word` 
    # node of query into `Group()` of morphologically equivalent `Word`s
    query_processor = QueryProcessor(transformers=[MorphyTransformer()])
    query = 'three dogs'
    language = detect_language(query)
    processed_query = query_processor.process(query, language=language)
    summa_query = processed_query.to_summa_query()
    assert summa_query == {
        'boolean': {'subqueries': [
            {'occur': 'should', 'query': {'match': {'value': 'three'}}},
            {'occur': 'should', 'query': {'match': {'value': 'dogs'}}},
            {'occur': 'should', 'query': {
                'boost': {'query': {'match': {'value': 'dog'}}, 'score': '0.85000'}}
            }
        ]}
    }

    # Collectors are described at https://izihawa.github.io/summa/collectors
    # Here we are requesting `TopDocs` collector with limit 10 that means 
    # that top-10 documents will be returned
    results = await client.search(
        summa_query,
        collectors=[{'top_docs': {'limit': 10}}],
    )
    return results

asyncio.run(main())
```
