---
title: Python
parent: APIs
---

# Aiosumma
Aiosumma consists of two parts: Python client for GRPC API and query library. While Python client for GRPC API
is pretty plain, query library provides tools for extending queries and deriving intents through various tools

## Query Library
```python
import asyncio

from aiosumma import SummaClient

async def main():
    # Create client and connect to the endpoint
    client = SummaClient('localhost:8082')
    await client.start_and_wait()
    
    # Collectors are described at https://izihawa.github.io/summa/core/collectors
    # Here we are requesting `TopDocs` collector with limit 10 that means 
    # that top-10 documents will be returned
    results = await client.search([{
        'index_alias': 'books',
        'query': {
            'boolean': {'subqueries': [
                {'occur': 'should', 'query': {'match': {'value': 'three'}}},
                {'occur': 'should', 'query': {'match': {'value': 'dogs'}}},
                {'occur': 'should', 'query': {
                    'boost': {'query': {'match': {'value': 'dog'}}, 'score': '0.65'}}
                }]}
        },
        'collectors': [{'top_docs': {'limit': 10}}]
    }])
    return results

asyncio.run(main())
```
