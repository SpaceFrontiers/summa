import os

import fire
from library.clicolor import Colored
from library.configurator import Configurator
from utils.file import mkdir_p


class Commander:
    def __init__(self, schema_path, index_path):
        self.schema_path = schema_path
        self.index_path = index_path

    def init(self):
        if not os.path.exists(self.schema_path):
            print(f"Creating schema directory {self.schema_path}... ", end='')
            mkdir_p(self.schema_path)
            print(Colored.Ok('ok'))


if __name__ == '__main__':
    config = Configurator(['summa/config.yaml'])
    commander = Commander(
        schema_path=config['search_engine']['schema_path'],
        index_path=config['search_engine']['index_path'],
    )
    fire.Fire(commander)
