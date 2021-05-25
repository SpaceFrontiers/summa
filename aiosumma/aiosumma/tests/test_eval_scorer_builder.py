from aiosumma.eval_scorer_builder import EvalScorerBuilder
from datetime import timedelta


def test_eval_scorer_builder():
    e = EvalScorerBuilder()
    q = e.add_exp_decay(
        field_name='x',
        origin=100,
        scale=timedelta(seconds=50),
        offset=timedelta(seconds=10),
        decay=0.5,
    )
    assert q.build() == ''
