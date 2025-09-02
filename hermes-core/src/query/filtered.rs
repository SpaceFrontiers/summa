//! Common eligibility without filter scores, ordinals, or pre-filter top-k.
use super::{DocBitset, Query, Scorer, ScorerFuture, ScorerOptions};
use crate::segment::SegmentReader;
use crate::{Error, Result};
use std::sync::Arc;

#[derive(Clone)]
pub struct FilteredQuery {
    query: Arc<dyn Query>,
    filters: Vec<Arc<dyn Query>>,
}
impl FilteredQuery {
    pub fn new(query: Arc<dyn Query>, filters: Vec<Arc<dyn Query>>) -> Self {
        Self { query, filters }
    }
    fn validate(&self, reader: &SegmentReader) -> Result<()> {
        if self.filters.len() > 64 || reader.num_docs() as usize > 128 * 1024 * 1024 {
            return Err(Error::Query(
                "common filter exceeds the 64-clause/16 MiB bitmap budget".into(),
            ));
        }
        Ok(())
    }
}
impl std::fmt::Display for FilteredQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} with {} eligibility filters",
            self.query,
            self.filters.len()
        )
    }
}
fn intersect(combined: &mut Option<DocBitset>, next: DocBitset) {
    if let Some(existing) = combined {
        existing.intersect_with(&next);
    } else {
        *combined = Some(next);
    }
}
fn enumerate_filter(mut scorer: Box<dyn Scorer + '_>, num_docs: u32) -> DocBitset {
    let mut bits = DocBitset::new(num_docs);
    while scorer.doc() != crate::TERMINATED {
        bits.set(scorer.doc());
        scorer.advance();
    }
    bits
}
// Native materializable filters stream their matches directly into a bitset.
// The portable fallback uses ordinary scorers only on bounded segments, never
// allocating a corpus-sized top-k on a production-scale legacy field.
fn fallback_limit(reader: &SegmentReader) -> Result<usize> {
    if reader.num_docs() as usize > super::MAX_FUSION_CANDIDATE_SLOTS {
        return Err(Error::Query("common filter cannot be materialized by this backend on this segment; use an indexed text/phrase/fast-field filter with sync support".into()));
    }
    Ok(reader.num_docs() as usize)
}
fn filtered<'a>(
    scorer: Box<dyn Scorer + 'a>,
    bits: Option<Arc<DocBitset>>,
) -> Box<dyn Scorer + 'a> {
    match bits {
        None => scorer,
        Some(bits) => Box::new(super::PredicatedScorer::new(
            scorer,
            vec![Box::new(move |doc| bits.contains(doc))],
            vec![],
            vec![],
        )),
    }
}
impl Query for FilteredQuery {
    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> super::CountFuture<'a> {
        self.query.count_estimate(reader)
    }
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        self.scorer_with_options(reader, limit, ScorerOptions::with_positions())
    }
    fn scorer_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        mut options: ScorerOptions,
    ) -> ScorerFuture<'a> {
        let this = self.clone();
        Box::pin(async move {
            this.validate(reader)?;
            let mut combined = None;
            for filter in &this.filters {
                let bits = match filter.as_doc_bitset(reader) {
                    Some(bits) => bits,
                    None => enumerate_filter(
                        filter
                            .scorer_with_options(
                                reader,
                                fallback_limit(reader)?,
                                ScorerOptions::default(),
                            )
                            .await?,
                        reader.num_docs(),
                    ),
                };
                intersect(&mut combined, bits);
            }
            if let (Some(bits), Some(outer)) = (&mut combined, &options.eligibility) {
                bits.intersect_with(outer);
            }
            options.eligibility = combined.map(Arc::new).or(options.eligibility);
            let bits = options.eligibility.clone();
            let scorer = this
                .query
                .scorer_with_options(reader, limit, options)
                .await?;
            Ok(filtered(scorer, bits))
        })
    }
    #[cfg(feature = "sync")]
    fn scorer_sync_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        mut options: ScorerOptions,
    ) -> Result<Box<dyn Scorer + 'a>> {
        self.validate(reader)?;
        let mut combined = None;
        for filter in &self.filters {
            let bits = match filter.as_doc_bitset(reader) {
                Some(bits) => bits,
                None => enumerate_filter(
                    filter.scorer_sync_with_options(
                        reader,
                        fallback_limit(reader)?,
                        ScorerOptions::default(),
                    )?,
                    reader.num_docs(),
                ),
            };
            intersect(&mut combined, bits);
        }
        if let (Some(bits), Some(outer)) = (&mut combined, &options.eligibility) {
            bits.intersect_with(outer);
        }
        options.eligibility = combined.map(Arc::new).or(options.eligibility);
        let bits = options.eligibility.clone();
        Ok(filtered(
            self.query
                .scorer_sync_with_options(reader, limit, options)?,
            bits,
        ))
    }
    fn decompose(&self) -> super::QueryDecomposition {
        self.query.decompose()
    }
    fn text_terms(&self, out: &mut Vec<(crate::Field, Vec<u8>)>) {
        self.query.text_terms(out);
        for filter in &self.filters {
            filter.text_terms(out);
        }
    }
}
