import datetime
import time


class QueryContext:
    def __init__(self, language=None):
        self.dois = []
        self.index_aliases = []
        self.language = language
        self.order_by = None
        self.query_point_of_time = time.time()
        self.explain = False

    def set_query_point_of_time(self, year):
        self.query_point_of_time = time.mktime(datetime.date(year, 7, 1).timetuple())
