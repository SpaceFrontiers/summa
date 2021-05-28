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
    client = SummaClient('localhost:8082')
    await client.start_and_wait()
    
    query_processor = QueryProcessor(transformers=[MorphyTransformer()])
    query = 'general astronomy'
    language = detect_language(query)
    processed_query = query_processor.process(query, language=language)
    assert processed_query == {}
    results = await client.search(
        processed_query.to_summa_query(),
        collectors=[{'top_docs': {'limit': 10}}],
    )
    assert results == []

asyncio.run(main())
```
