# aiocrossref

Asynchronous client for Summa API

## Example

```python
import asyncio

from aiosumma import SummaHttpClient

async def search(base_url, schema, query):
    client = SummaHttpClient(base_url)
    return await client.search(schema, query)

response = asyncio.get_event_loop().run_until_complete(search('0.0.0.0', 'schema', 'cookie recipe'))
```