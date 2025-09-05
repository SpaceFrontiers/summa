//! Common eligibility without filter scores, ordinals, or pre-filter top-k.
use super::{DocBitset, Query, Scorer, ScorerFuture, ScorerOptions};
use crate::segment::SegmentReader;
use crate::{Error, Result};
use std::sync::Arc;

pub(super) const MAX_FILTER_BITMAP_DOCS: usize = 128 * 1024 * 1024;

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
        if self.filters.len() > 64 || reader.num_docs() as usize > MAX_FILTER_BITMAP_DOCS {
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
/// Intersect complete filters, returning whether any document remains eligible.
fn intersect(combined: &mut Option<DocBitset>, next: DocBitset) -> bool {
    if let Some(existing) = combined {
        existing.intersect_with(&next);
    } else {
        *combined = Some(next);
    }
    combined.as_ref().unwrap().next_set_bit(0).is_some()
}
fn enumerate_filter(
    mut scorer: Box<dyn Scorer + '_>,
    num_docs: u32,
    options: &ScorerOptions,
) -> Option<DocBitset> {
    let mut bits = DocBitset::new(num_docs);
    let mut visited = 0usize;
    while scorer.doc() != crate::TERMINATED {
        if visited.is_multiple_of(1024) && options.stop_if_expired() {
            return None;
        }
        bits.set(scorer.doc());
        scorer.advance();
        visited += 1;
    }
    (!options.stop_if_expired()).then_some(bits)
}
fn no_candidates(options: &ScorerOptions) -> bool {
    options.stop_if_expired()
        || options
            .eligibility
            .as_ref()
            .is_some_and(|bits| bits.next_set_bit(0).is_none())
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
pub(super) fn filtered<'a>(
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
            if no_candidates(&options) {
                return Ok(Box::new(super::EmptyScorer) as Box<dyn Scorer>);
            }
            let filter_options = ScorerOptions {
                eligibility: None,
                collect_positions: false,
                ..options.without_threshold()
            };
            let mut combined = None;
            for filter in &this.filters {
                let bits = match filter_options.doc_bitset(filter.as_ref(), reader) {
                    Some(bits) => bits,
                    None => {
                        if filter_options.stop_if_expired() {
                            return Ok(Box::new(super::EmptyScorer) as Box<dyn Scorer>);
                        }
                        let Some(bits) = enumerate_filter(
                            filter
                                .scorer_with_options(
                                    reader,
                                    fallback_limit(reader)?,
                                    filter_options.clone(),
                                )
                                .await?,
                            reader.num_docs(),
                            &filter_options,
                        ) else {
                            return Ok(Box::new(super::EmptyScorer) as Box<dyn Scorer>);
                        };
                        bits
                    }
                };
                if filter_options.stop_if_expired() || !intersect(&mut combined, bits) {
                    return Ok(Box::new(super::EmptyScorer) as Box<dyn Scorer>);
                }
            }
            if let (Some(bits), Some(outer)) = (&mut combined, &options.eligibility) {
                bits.intersect_with(outer);
            }
            options.eligibility = combined.map(Arc::new).or(options.eligibility);
            if no_candidates(&options) {
                return Ok(Box::new(super::EmptyScorer) as Box<dyn Scorer>);
            }
            let bits = options.eligibility.clone();
            let scorer = this
                .query
                .scorer_with_options(reader, limit, options)
                .await?;
            Ok(filtered(scorer, bits))
        })
    }
    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        self.scorer_sync_with_options(reader, limit, ScorerOptions::with_positions())
    }
    #[cfg(feature = "sync")]
    fn scorer_sync_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        mut options: ScorerOptions,
    ) -> Result<Box<dyn Scorer + 'a>> {
        self.validate(reader)?;
        if no_candidates(&options) {
            return Ok(Box::new(super::EmptyScorer));
        }
        let filter_options = ScorerOptions {
            eligibility: None,
            collect_positions: false,
            ..options.without_threshold()
        };
        let mut combined = None;
        for filter in &self.filters {
            let bits = match filter_options.doc_bitset(filter.as_ref(), reader) {
                Some(bits) => bits,
                None => {
                    if filter_options.stop_if_expired() {
                        return Ok(Box::new(super::EmptyScorer));
                    }
                    let Some(bits) = enumerate_filter(
                        filter.scorer_sync_with_options(
                            reader,
                            fallback_limit(reader)?,
                            filter_options.clone(),
                        )?,
                        reader.num_docs(),
                        &filter_options,
                    ) else {
                        return Ok(Box::new(super::EmptyScorer));
                    };
                    bits
                }
            };
            if filter_options.stop_if_expired() || !intersect(&mut combined, bits) {
                return Ok(Box::new(super::EmptyScorer));
            }
        }
        if let (Some(bits), Some(outer)) = (&mut combined, &options.eligibility) {
            bits.intersect_with(outer);
        }
        options.eligibility = combined.map(Arc::new).or(options.eligibility);
        if no_candidates(&options) {
            return Ok(Box::new(super::EmptyScorer));
        }
        let bits = options.eligibility.clone();
        Ok(filtered(
            self.query
                .scorer_sync_with_options(reader, limit, options)?,
            bits,
        ))
    }
    fn decompose(&self) -> super::QueryDecomposition {
        if self.filters.is_empty() {
            self.query.decompose()
        } else {
            super::QueryDecomposition::Opaque
        }
    }
    fn lsp_decomposition(&self) -> super::QueryDecomposition {
        self.query.lsp_decomposition()
    }
    fn text_terms(&self, out: &mut Vec<(crate::Field, Vec<u8>)>) {
        self.query.text_terms(out);
        for filter in &self.filters {
            filter.text_terms(out);
        }
    }
}

#[cfg(all(test, feature = "native"))]
mod tests;
