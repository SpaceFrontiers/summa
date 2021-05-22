#[cfg(test)]
pub mod shortcuts {
    use crate::proto;

    pub fn top_docs_collector(limit: u32) -> proto::Collector {
        proto::Collector {
            collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
                limit,
                offset: 0,
                scorer: None,
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
            })),
        }
    }

    pub fn scored_doc(document: &str, score: f64, position: u32) -> proto::ScoredDocument {
        proto::ScoredDocument {
            document: document.to_owned(),
            score: Some(proto::Score {
                score: Some(proto::score::Score::F64Score(score)),
            }),
            position,
        }
    }

    pub fn top_docs_collector_result(scored_documents: Vec<proto::ScoredDocument>, has_next: bool) -> proto::CollectorResult {
        proto::CollectorResult {
            collector_result: Some(proto::collector_result::CollectorResult::TopDocs(proto::TopDocsCollectorResult {
                scored_documents,
                has_next,
            })),
        }
    }
}
