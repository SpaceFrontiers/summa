//! `From` trait implmentations used for creation of `proto::Score`

use crate::proto;

impl From<f32> for proto::Score {
    fn from(score: f32) -> Self {
        proto::Score {
            score: Some(proto::score::Score::F64Score(score.into())),
        }
    }
}

impl From<f64> for proto::Score {
    fn from(score: f64) -> Self {
        proto::Score {
            score: Some(proto::score::Score::F64Score(score)),
        }
    }
}

impl From<u64> for proto::Score {
    fn from(score: u64) -> Self {
        proto::Score {
            score: Some(proto::score::Score::U64Score(score)),
        }
    }
}
