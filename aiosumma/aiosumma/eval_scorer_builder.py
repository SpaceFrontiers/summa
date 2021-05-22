from datetime import timedelta

from summa.proto import search_service_pb2 as search_service_pb


class EvalScorerBuilder:
    def __init__(self):
        self.ops = ['original_score']

    def add_point_fade(self, field_name, point_of_time, interval: timedelta, baseline):
        self.ops.append(f'point_fade({field_name}, {point_of_time}, {interval.total_seconds()}, {baseline})')

    def add_fastsigm(self, field_name, alpha):
        self.ops.append(f'fastsigm({field_name}, {alpha})')

    def build(self):
        if not self.ops:
            return search_service_pb.Scorer()
        return search_service_pb.Scorer(eval_expr=' * '.join(self.ops))
