import asyncio
import sys
import traceback

import fire

from aiosumma import SummaClient


async def client_cli(endpoint):
    try:
        client = SummaClient(endpoint=endpoint, connection_timeout=3.0)
        await client.start_and_wait()
        return client.get_interface()
    except asyncio.exceptions.TimeoutError:
        # ToDo: process exception through fire.core.FireError
        print(traceback.format_exc())
        sys.exit(1)


def main():
    fire.Fire(client_cli, name='summa-client')


if __name__ == '__main__':
    main()
