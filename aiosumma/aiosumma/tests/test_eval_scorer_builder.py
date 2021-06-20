from datetime import timedelta

from aiosumma.eval_scorer_builder import EvalScorerBuilder


def test_eval_scorer_builder():
    e = EvalScorerBuilder()
    q = e.add_exp_decay(
        field_name='x',
        origin=100,
        scale=timedelta(seconds=50),
        offset=timedelta(seconds=10),
        decay=0.5,
    )
    assert q.build().eval_expr == 'original_score * e() ^ ((log(e(), 0.5) / 50.0) * max(0, abs(x - 100) - 10.0))'
