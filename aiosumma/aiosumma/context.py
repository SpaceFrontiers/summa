import datetime
import time
from typing import (
    Dict,
    Optional,
)


class QueryContext:
    def __init__(self, languages: Optional[Dict[str, float]] = None, query_language: Optional[str] = None):
        self.dois = []
        self.index_aliases = []
        self.languages = languages
        self.query_language = query_language
        self.order_by = None
        self.query_point_of_time = time.time()
        self.explain = False
        self.has_invalid_fields = False

    def set_query_point_of_time(self, year):
        self.query_point_of_time = time.mktime(datetime.date(year, 7, 1).timetuple())

    def blank_query(self):
        if self.has_invalid_fields:
            return {'empty': {}}
        return {'all': {}}
