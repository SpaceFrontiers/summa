import datetime
import time


class QueryContext:
    def __init__(self, language=None):
        self.dois = []
        self.index_aliases = []
        self.is_forced_clean = False
        self.is_exploration = True
        self.language = language
        self.order_by = None
        self.query_point_of_time = time.time()

    def set_query_point_of_time(self, year):
        self.query_point_of_time = time.mktime(datetime.date(year, 7, 1).timetuple())

    def with_user_intent(self):
        return self.is_exploration and not self.is_forced_clean
