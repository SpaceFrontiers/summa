//! Read-only diagnostics: exact collector versus exhaustive scorer enumeration.
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use summa_core::directories::MmapDirectory;
use summa_core::query::{Collector, CountCollector, TopKCollector, collect_segment};
use summa_core::tokenizer::{Purpose, TokenizerRegistry};
use summa_core::{Index, IndexConfig};

pub async fn run(path: &Path, input: &Path, config: IndexConfig) -> Result<()> {
    let index = Index::open(MmapDirectory::new(path), config).await?;
    let searcher = index.reader().await?.searcher().await?;
    let field = searcher
        .schema()
        .get_field("body")
        .context("missing body")?;
    let id = searcher.schema().get_field("id").context("missing id")?;
    let tokenizer = TokenizerRegistry::new()
        .get(
            searcher
                .schema()
                .get_field_entry(field)
                .and_then(|entry| entry.tokenizer.as_deref())
                .unwrap_or("default"),
        )
        .context("invalid tokenizer")?;
    let parser = searcher.query_parser();
    let source = std::io::BufReader::new(std::fs::File::open(input)?);
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in source.lines() {
        let mut value: Value = serde_json::from_str(&line?)?;
        if let Some(text) = value.get("text").and_then(Value::as_str) {
            let tokens: Vec<_> = tokenizer
                .tokenize_with(text, None, Purpose::Exact)
                .into_iter()
                .map(|t| json!({"term":t.text,"position":t.position}))
                .collect();
            writeln!(output, "{}", json!({"text":text,"tokens":tokens}))?;
            continue;
        }
        value["limit"] = json!(0);
        let request =
            super::query::Envelope::from_value(value.clone())?.parse(&parser, field, &tokenizer)?;
        let check_ranked = value["audit_ranked"].as_bool().unwrap_or(false);
        anyhow::ensure!(
            !check_ranked || searcher.num_segments() == 1,
            "rank audit requires one segment"
        );
        let mut oracle = TopKCollector::new(100);
        let mut optimized = 0;
        let mut exhaustive = 0u64;
        let mut sample = Vec::new();
        for segment in searcher.segment_readers() {
            let mut collector = CountCollector::new();
            collect_segment(segment, request.query.as_ref(), &mut collector).await?;
            optimized += collector.count();
            let column = segment.fast_field(id.0).context("missing id column")?;
            let mut audit = ExhaustiveAudit {
                top: &mut oracle,
                ranked: check_ranked,
                count: 0,
                sample: Vec::new(),
            };
            if request.query.should_children().is_some() {
                // A zero limit does not make a union scorer exhaustive. The
                // public complete collector path supplies that contract.
                collect_segment(segment, request.query.as_ref(), &mut audit).await?;
            } else {
                let mut scorer = request.query.scorer_sync(segment, 0)?;
                while scorer.doc() != summa_core::structures::TERMINATED {
                    audit.collect(
                        scorer.doc(),
                        if check_ranked { scorer.score() } else { 0.0 },
                        &[],
                    );
                    scorer.advance();
                }
            }
            exhaustive += audit.count;
            for doc in audit.sample {
                if sample.len() == 10 {
                    break;
                }
                sample.push(
                    column
                        .get_text(doc)
                        .context("missing external id")?
                        .to_owned(),
                );
            }
        }
        let ranked = if check_ranked {
            let actual = searcher
                .search_with_offset_and_count_sync(request.query.as_ref(), 100, 0)?
                .0;
            let expected = oracle.into_sorted_results();
            anyhow::ensure!(
                actual.len() == expected.len()
                    && actual.iter().zip(&expected).all(
                        |(a, b)| a.doc_id == b.doc_id && a.score.to_bits() == b.score.to_bits()
                    ),
                "ranked results differ from exhaustive oracle"
            );
            let column = searcher.segment_readers()[0]
                .fast_field(id.0)
                .context("missing id column")?;
            actual.iter().map(|hit| Ok(json!({"id": column.get_text(hit.doc_id).context("missing id")?, "score_bits": hit.score.to_bits()}))).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        writeln!(
            output,
            "{}",
            json!({"query":value["query"],"class":value["class"],
            "plan":request.query.to_string(),"optimized":optimized,"exhaustive":exhaustive,"sample":sample,"ranked":ranked})
        )?;
        output.flush()?;
    }
    Ok(())
}

/// Keep callbacks exhaustive: do not advertise a ranked count limit or accept
/// metadata-only counts. Retained scratch is the top-100 heap plus ten IDs.
struct ExhaustiveAudit<'a> {
    top: &'a mut TopKCollector,
    ranked: bool,
    count: u64,
    sample: Vec<u32>,
}

impl Collector for ExhaustiveAudit<'_> {
    fn collect(
        &mut self,
        doc: u32,
        score: f32,
        _: &[(u32, Vec<summa_core::query::ScoredPosition>)],
    ) {
        self.count += 1;
        if self.sample.len() < 10 {
            self.sample.push(doc);
        }
        if self.ranked {
            self.top.collect(doc, score, &[]);
        }
    }
    fn needs_scores(&self) -> bool {
        self.ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use summa_core::query::{BooleanQuery, TermQuery};
    use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
    use summa_core::{Document, RamDirectory, SchemaBuilder};

    #[tokio::test]
    async fn union_audit_counts_every_match_even_with_a_small_top_k_heap() {
        let mut schema = SchemaBuilder::default();
        let body = schema.add_text_field("body", true, false);
        let schema = Arc::new(schema.build());
        let directory = RamDirectory::new();
        let mut builder =
            SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
        for text in ["alpha", "beta", "alpha beta"] {
            let mut doc = Document::new();
            doc.add_text(body, text);
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&directory, id, None).await.unwrap();
        let reader = SegmentReader::open(&directory, id, schema, 16)
            .await
            .unwrap();
        let query = BooleanQuery::new()
            .should(TermQuery::text(body, "alpha"))
            .should(TermQuery::text(body, "beta"));
        let mut top = TopKCollector::new(1);
        let mut audit = ExhaustiveAudit {
            top: &mut top,
            ranked: true,
            count: 0,
            sample: Vec::new(),
        };
        collect_segment(&reader, &query, &mut audit).await.unwrap();
        assert_eq!(audit.count, 3);
        assert_eq!(audit.sample, [0, 1, 2]);
        assert_eq!(top.into_sorted_results()[0].doc_id, 2);
    }
}
