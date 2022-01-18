use std::cmp::Ordering;

use crate::proto;

impl PartialOrd for proto::score::Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = match self {
            proto::score::Score::F64Score(score) => *score,
            proto::score::Score::U64Score(score) => *score as f64,
        };
        let b = match other {
            proto::score::Score::F64Score(score) => *score,
            proto::score::Score::U64Score(score) => *score as f64,
        };
        a.partial_cmp(&b)
    }
}

impl PartialOrd for proto::Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl PartialOrd for proto::ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl proto::CollectorOutput {
    pub fn as_top_docs(&self) -> Option<&proto::TopDocsCollectorOutput> {
        if let Some(proto::collector_output::CollectorOutput::TopDocs(top_docs)) = &self.collector_output {
            Some(top_docs)
        } else {
            None
        }
    }

    pub fn as_count(&self) -> Option<&proto::CountCollectorOutput> {
        if let Some(proto::collector_output::CollectorOutput::Count(count)) = &self.collector_output {
            Some(count)
        } else {
            None
        }
    }
}

pub(crate) mod shortcuts {
    use std::collections::HashMap;

    use crate::proto;

    pub fn top_docs_collector(limit: u32) -> proto::Collector {
        proto::Collector {
            collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
                limit,
                offset: 0,
                scorer: None,
                snippets: HashMap::new(),
                explain: false,
                fields: Vec::new(),
            })),
        }
    }

    pub fn top_docs_collector_with_eval_expr(limit: u32, eval_expr: &str) -> proto::Collector {
        proto::Collector {
            collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
                limit,
                offset: 0,
                scorer: Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::EvalExpr(eval_expr.to_owned())),
                }),
                snippets: HashMap::new(),
                explain: false,
                fields: Vec::new(),
            })),
        }
    }

    pub fn scored_doc(document: &str, score: f64, position: u32) -> proto::ScoredDocument {
        proto::ScoredDocument {
            index_alias: "test_index".to_string(),
            document: document.to_owned(),
            score: Some(proto::Score {
                score: Some(proto::score::Score::F64Score(score)),
            }),
            position,
            snippets: HashMap::new(),
        }
    }

    pub fn top_docs_collector_output(scored_documents: Vec<proto::ScoredDocument>, has_next: bool) -> proto::CollectorOutput {
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
                scored_documents,
                has_next,
            })),
        }
    }
}
