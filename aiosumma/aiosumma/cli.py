import fire
from aiosumma.client import SummaClient


async def cli(endpoint):
    client = SummaClient(endpoint=endpoint)
    await client.start_and_wait()
    return client.get_interface()


def main():
    fire.Fire(cli)


if __name__ == '__main__':
    main()
