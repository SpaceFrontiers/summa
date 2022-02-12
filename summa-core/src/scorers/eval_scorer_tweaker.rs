use tantivy::collector::{ScoreSegmentTweaker, ScoreTweaker};
use tantivy::{DocId, SegmentReader};

use crate::scorers::{EvalScorer, SegmentEvalScorer};

pub(crate) struct EvalScorerSegmentScoreTweaker {
    segment_eval_scorer: SegmentEvalScorer,
}

impl EvalScorerSegmentScoreTweaker {
    pub fn new(segment_eval_scorer: SegmentEvalScorer) -> Self {
        EvalScorerSegmentScoreTweaker { segment_eval_scorer }
    }
}

impl ScoreSegmentTweaker<f64> for EvalScorerSegmentScoreTweaker {
    fn score(&mut self, doc: DocId, score: f32) -> f64 {
        self.segment_eval_scorer.score(doc, score)
    }
}

pub(crate) struct EvalScorerTweaker {
    eval_scorer_seed: EvalScorer,
}

impl EvalScorerTweaker {
    pub fn new(eval_scorer_seed: EvalScorer) -> Self {
        EvalScorerTweaker { eval_scorer_seed }
    }
}

#[async_trait]
impl ScoreTweaker<f64> for EvalScorerTweaker {
    type Child = EvalScorerSegmentScoreTweaker;

    fn segment_tweaker(&self, segment_reader: &SegmentReader) -> tantivy::Result<Self::Child> {
        Ok(EvalScorerSegmentScoreTweaker::new(
            self.eval_scorer_seed.get_for_segment_reader(segment_reader).expect("Wrong eval expression"),
        ))
    }

    async fn segment_tweaker_async(&self, segment_reader: &SegmentReader) -> tantivy::Result<Self::Child> {
        Ok(EvalScorerSegmentScoreTweaker::new(
            self.eval_scorer_seed
                .get_for_segment_reader_async(segment_reader)
                .await
                .expect("Wrong eval expression"),
        ))
    }
}
