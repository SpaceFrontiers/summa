import logging

from izihawa_utils.exceptions import BaseError


class ParseError(BaseError):
    code = 'parse_error'
    level = logging.WARNING
