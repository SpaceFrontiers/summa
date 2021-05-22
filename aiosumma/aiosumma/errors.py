from izihawa_utils.exceptions import BaseError


class UnsupportedQueryError(BaseError):
    code = 'unsupported_query_error'
