import os
import sys

import pytest

if __name__ == '__main__':
    sys.exit(pytest.main([os.path.dirname(__file__), '-vvv', '-W', 'ignore::DeprecationWarning']))
