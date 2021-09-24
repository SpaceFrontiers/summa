#[cfg(test)]
pub mod shortcuts {
    use crate::proto;
    use std::collections::HashMap;

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
            index_alias: "index".to_string(),
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
