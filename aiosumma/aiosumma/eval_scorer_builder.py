from datetime import timedelta

from summa.proto import search_service_pb2 as search_service_pb


class EvalScorerBuilder:
    def __init__(self):
        self.ops = ['original_score']

    def add_gauss(self, field_name, point_of_time, offset: timedelta, interval: timedelta, decay):
        self.ops.append(f'gauss({field_name}, {point_of_time}, {offset.total_seconds()}, '
                        f'sigma_squared({interval.total_seconds()}, {decay}))')

    def add_fastsigm(self, field_name, alpha):
        self.ops.append(f'fastsigm({field_name}, {alpha})')

    def build(self):
        if not self.ops:
            return search_service_pb.Scorer()
        return search_service_pb.Scorer(eval_expr=' * '.join(self.ops))
