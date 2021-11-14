#!/usr/bin/env python3

import asyncio
import sys

import fire
from termcolor import colored

from aiosumma import SummaClient


async def client_cli(endpoint):
    try:
        client = SummaClient(endpoint=endpoint, connection_timeout=3.0)
        await client.start()
        return client.get_interface()
    except asyncio.exceptions.TimeoutError:
        # ToDo: process exception through fire.core.FireError
        print(f"{colored('ERROR', 'red')}: {endpoint} timeout", file=sys.stderr)
        sys.exit(1)


def main():
    fire.Fire(client_cli, name='summa-client')


if __name__ == '__main__':
    main()
