from datetime import timedelta

from summa.proto import search_service_pb2


class EvalScorerBuilder:
    def __init__(self):
        self.ops = ['original_score']

    def add_gauss_decay(self, field_name, origin, scale: timedelta, offset: timedelta, decay):
        ss = f'{scale.total_seconds()}^2/log(e(), {decay})'
        self.ops.append(f'e()^(-(max(0, abs({field_name} - {origin}) - {offset.total_seconds()})^2)/({ss}))')
        return self

    def add_exp_decay(self, field_name, origin, scale: timedelta, offset: timedelta, decay):
        lamb = f'log(e(), {decay}) / {scale.total_seconds()}'
        self.ops.append(f'e() ^ (({lamb}) * max(0, abs({field_name} - {origin}) - {offset.total_seconds()}))')
        return self

    def add_linear_decay(self, field_name, origin, scale: timedelta, offset: timedelta, decay):
        s = f'{scale.total_seconds()}/(1.0 - {decay})'
        self.ops.append(f'max(0, ({s} - max(0, abs({field_name} - {origin}) - {offset.total_seconds()}))/({s}))')
        return self

    def add_fastsigm(self, field_name, alpha):
        self.ops.append(f'fastsigm({field_name}, {alpha})')
        return self

    def build(self) -> search_service_pb2.Scorer:
        if not self.ops:
            return {}
        return search_service_pb2.Scorer(eval_expr=' * '.join(self.ops))
