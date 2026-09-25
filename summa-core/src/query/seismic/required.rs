//! Complete membership and document-order scoring for required clauses.
use super::scoring::{PreparedQuery, document_ordinals};
use crate::query::{MultiValueCombiner, ScorerOptions, SparseTermQueryInfo};
use crate::segment::VectorOrdinals;
use crate::segment::seismic::SeismicIndex;
use crate::{Error, Result};

/// Complete sparse membership for Boolean filters, independent of nomination
/// and top-k limits. No payload score or ordinal list is allocated.
pub(in crate::query) fn membership(
    index: &SeismicIndex,
    docs: u32,
    terms: &[(u32, f32)],
    options: &ScorerOptions,
) -> Option<crate::query::DocBitset> {
    if docs as usize > crate::query::filtered::MAX_FILTER_BITMAP_DOCS {
        return None;
    }
    let mut dimensions: Vec<_> = terms
        .iter()
        .filter(|(_, weight)| *weight != 0.0)
        .map(|&(dimension, _)| dimension)
        .collect();
    dimensions.sort_unstable();
    dimensions.dedup();
    let mut bits = crate::query::DocBitset::new(docs);
    for row in 0..index.len() {
        if row.is_multiple_of(1024) && options.stop_if_expired() {
            return None;
        }
        if index
            .vector(row)
            .iter()
            .any(|(dimension, _)| dimensions.binary_search(&dimension).is_ok())
        {
            bits.set(index.key(row).doc);
        }
    }
    (!options.stop_if_expired()).then_some(bits)
}

/// Required scoring clauses expose complete membership in document order.
/// Scorer iteration cannot return errors, so construction admits the immutable
/// score stream once; the second pass needs only one document's ordinal scratch.
pub(in crate::query) fn required_scorer<'a>(
    reader: &'a crate::segment::SegmentReader,
    field: crate::Field,
    infos: &[SparseTermQueryInfo],
    options: &ScorerOptions,
) -> Result<Option<Box<dyn crate::query::Scorer + 'a>>> {
    let Some(index) = reader.seismic_index(field) else {
        return Ok(None);
    };
    let Some(info) = infos.first() else {
        return Ok(Some(Box::new(crate::query::EmptyScorer)));
    };
    let query = PreparedQuery::new(infos.iter().map(|i| (i.dim_id, i.weight)).collect());
    let mut scorer = RequiredScorer {
        reader,
        index,
        field,
        query,
        combiner: info.combiner,
        options: options.clone(),
        next_row: 0,
        doc: crate::TERMINATED,
        score: 0.0,
        ordinals: VectorOrdinals::new(),
        matched_ordinals: VectorOrdinals::new(),
    };
    while scorer.next_document()? != crate::TERMINATED {}
    scorer.next_row = 0;
    scorer.next_document()?;
    Ok(Some(Box::new(scorer)))
}
struct RequiredScorer<'a> {
    reader: &'a crate::segment::SegmentReader,
    index: &'a SeismicIndex,
    field: crate::Field,
    query: PreparedQuery,
    combiner: MultiValueCombiner,
    options: ScorerOptions,
    next_row: u32,
    doc: u32,
    score: f32,
    ordinals: VectorOrdinals,
    matched_ordinals: VectorOrdinals,
}
impl RequiredScorer<'_> {
    fn next_document(&mut self) -> Result<u32> {
        while self.next_row < self.index.len() {
            if self.options.stop_if_expired() {
                break;
            }
            let doc = self.index.key(self.next_row).doc;
            while self.next_row < self.index.len() && self.index.key(self.next_row).doc == doc {
                self.next_row += 1;
            }
            if !self.reader.is_alive(doc)
                || self
                    .options
                    .eligibility
                    .as_ref()
                    .is_some_and(|bits| !bits.contains(doc))
            {
                continue;
            }
            document_ordinals(
                self.index,
                doc,
                &self.query,
                &mut self.ordinals,
                self.options
                    .collect_positions
                    .then_some(&mut self.matched_ordinals),
            )?;
            if self.ordinals.is_empty() {
                continue;
            }
            self.score = self.combiner.combine(&self.ordinals);
            if !self.score.is_finite() {
                return Err(Error::Query("Seismic combined score overflow".into()));
            }
            self.doc = doc;
            return Ok(doc);
        }
        self.doc = crate::TERMINATED;
        Ok(self.doc)
    }
}
impl crate::query::DocSet for RequiredScorer<'_> {
    fn doc(&self) -> u32 {
        self.doc
    }
    fn advance(&mut self) -> u32 {
        self.next_document()
            .expect("immutable sparse scores admitted at scorer construction")
    }
    fn seek(&mut self, target: u32) -> u32 {
        while self.doc < target {
            self.advance();
        }
        self.doc
    }
    fn size_hint(&self) -> u32 {
        self.index.len().saturating_sub(self.next_row)
    }
}
impl crate::query::Scorer for RequiredScorer<'_> {
    fn score(&self) -> f32 {
        self.score
    }
    fn matched_positions(&self) -> Option<crate::query::MatchedPositions> {
        (self.doc != crate::TERMINATED).then(|| {
            vec![(
                self.field.0,
                self.matched_ordinals
                    .iter()
                    .map(|&(ordinal, score)| crate::query::ScoredPosition::new(ordinal, score))
                    .collect(),
            )]
        })
    }
}
