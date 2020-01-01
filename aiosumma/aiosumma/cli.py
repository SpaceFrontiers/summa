import fire
from aiokit.utils import sync_fu
from aiosumma.client import SummaHttpClient


async def search(base_url, schema, query):
    async with SummaHttpClient(base_url=base_url) as c:
        return await c.search(schema, query)


def main():
    fire.Fire({
        'search': sync_fu(search),
    })


if __name__ == '__main__':
    main()
