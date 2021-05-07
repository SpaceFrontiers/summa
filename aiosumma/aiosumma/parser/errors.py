import logging

from izihawa_utils.exceptions import BaseError


class UnsupportedQueryError(BaseError):
    code = 'unsupported_query_error'


class ParseError(BaseError):
    code = 'parse_error'
    level = logging.WARNING
