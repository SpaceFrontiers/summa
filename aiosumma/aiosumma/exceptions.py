import logging

from aiobaseclient.exceptions import (
    BaseError,
    TemporaryError,
)


class InvalidSyntaxError(BaseError):
    code = 'invalid_syntax_error'
    level = logging.WARNING

    def __init__(self, query, page):
        super().__init__(query=query, page=page)


class QueryTimeoutError(BaseError):
    code = 'query_timeout_error'
    level = logging.WARNING


class TimeoutError(TemporaryError):
    code = 'timeout_error'
    level = logging.WARNING
