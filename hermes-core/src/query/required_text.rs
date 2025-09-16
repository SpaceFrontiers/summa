//! Complete document membership on reordered chunk maps, with bounded scratch.

use std::sync::Arc;

use super::docset::DocSet;
use super::{
    Bm25Params, DocBitset, EmptyScorer, MatchedPositions, ScoredPosition, Scorer, SharedThreshold,
};
use crate::segment::SegmentReader;
use crate::segment::chunk_map::ChunkMap;
use crate::structures::{BlockPostingList, TERMINATED};
use crate::{DocId, Field, Score};

pub(super) fn scorer<'a>(
    postings: Vec<(BlockPostingList, f32)>,
    avg_len: f32,
    reader: &'a SegmentReader,
    field: Field,
    budget: Option<SharedThreshold>,
    eligibility: Option<Arc<DocBitset>>,
) -> crate::Result<Box<dyn Scorer + 'a>> {
    if budget
        .as_ref()
        .is_some_and(SharedThreshold::stop_if_expired)
    {
        return Ok(Box::new(EmptyScorer));
    }
    if reader.num_docs() as usize > super::filtered::MAX_FILTER_BITMAP_DOCS {
        return Err(crate::Error::Query(
            "required text exceeds the 16 MiB document bitmap budget".into(),
        ));
    }
    let map = reader.chunk_map(field).ok_or_else(|| {
        crate::Error::Corruption("required chunked text lacks a chunk map".into())
    })?;
    if !map.has_logical_addressing() {
        return Err(crate::Error::Query(
            "legacy reordered text needs explicit Reorder to upgrade its chunk map for required matching"
                .into(),
        ));
    }
    let mut matches = DocBitset::new(reader.num_docs());
    for (posting, _) in &postings {
        let mut cursor = posting.iterator();
        let mut visited = 0usize;
        while cursor.doc() != TERMINATED {
            if visited.is_multiple_of(1024)
                && budget
                    .as_ref()
                    .is_some_and(SharedThreshold::stop_if_expired)
            {
                return Ok(Box::new(EmptyScorer));
            }
            let doc = map.doc_id(cursor.doc());
            if eligibility.as_ref().is_none_or(|bits| bits.contains(doc)) {
                matches.set(doc);
            }
            cursor.advance();
            visited += 1;
        }
    }
    if budget
        .as_ref()
        .is_some_and(SharedThreshold::stop_if_expired)
    {
        return Ok(Box::new(EmptyScorer));
    }
    let mut scorer = RequiredTextScorer {
        postings,
        avg_len,
        map: map.clone(),
        field,
        budget,
        matches,
        params: Bm25Params::for_field(reader.schema(), field),
        current: TERMINATED,
        score: 0.0,
        slots: Vec::new(),
        decode: Default::default(),
    };
    scorer.position(0);
    Ok(Box::new(scorer))
}

struct RequiredTextScorer {
    postings: Vec<(BlockPostingList, f32)>,
    avg_len: f32,
    map: ChunkMap,
    field: Field,
    budget: Option<SharedThreshold>,
    matches: DocBitset,
    params: Bm25Params,
    current: DocId,
    score: Score,
    slots: Vec<(u32, u16, Option<f32>)>,
    decode: crate::structures::postings::PostingDecodeScratch,
}

impl RequiredTextScorer {
    fn position(&mut self, target: DocId) {
        self.score = 0.0;
        self.slots.clear();
        self.current = self.matches.next_set_bit(target).unwrap_or(TERMINATED);
        if self.doc() == TERMINATED {
            self.current = TERMINATED;
            return;
        }
        self.slots.extend(
            self.map
                .slots_for_document(self.current)
                .map(|(ordinal, slot)| (slot, ordinal, None)),
        );
        self.slots.sort_unstable_by_key(|&(slot, _, _)| slot);
        let Some(&(first, _, _)) = self.slots.first() else {
            return;
        };
        for (posting, idf) in &self.postings {
            let mut cursor = posting
                .clone()
                .into_candidate_iterator(first, &mut self.decode);
            for (index, (slot, _, score)) in self.slots.iter_mut().enumerate() {
                if index.is_multiple_of(1024)
                    && self
                        .budget
                        .as_ref()
                        .is_some_and(SharedThreshold::stop_if_expired)
                {
                    cursor.recycle(&mut self.decode);
                    self.current = TERMINATED;
                    self.score = 0.0;
                    self.slots.clear();
                    return;
                }
                if cursor.seek(*slot) == *slot {
                    *score = Some(
                        score.unwrap_or(0.0)
                            + self.params.score(
                                cursor.term_freq() as f32,
                                *idf,
                                self.map.bm25_length(*slot) as f32,
                                self.avg_len,
                            ),
                    );
                }
            }
            cursor.recycle(&mut self.decode);
        }
        self.score = self
            .slots
            .iter()
            .filter_map(|(_, _, score)| *score)
            .fold(0.0, f32::max);
        self.slots.sort_unstable_by_key(|&(_, ordinal, _)| ordinal);
    }
}

impl DocSet for RequiredTextScorer {
    fn doc(&self) -> DocId {
        if self
            .budget
            .as_ref()
            .is_some_and(SharedThreshold::stop_if_expired)
        {
            TERMINATED
        } else {
            self.current
        }
    }
    fn advance(&mut self) -> DocId {
        if self.doc() != TERMINATED {
            self.position(self.current + 1);
        }
        self.doc()
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc() != TERMINATED && target > self.current {
            self.position(target);
        }
        self.doc()
    }
    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for RequiredTextScorer {
    fn score(&self) -> Score {
        self.score
    }
    fn matched_positions(&self) -> Option<MatchedPositions> {
        if self.doc() == TERMINATED || self.map.is_document_map() {
            return None;
        }
        Some(vec![(
            self.field.0,
            self.slots
                .iter()
                .filter_map(|&(_, ordinal, score)| {
                    score.map(|score| ScoredPosition::new(u32::from(ordinal), score))
                })
                .collect(),
        )])
    }
}

/// Translate complete plain-text traversal into logical document order. Only
/// membership is materialized; scores and token positions stay with the original
/// scorer and are probed using its retained, bounded physical cursor buffers.
pub(super) fn mapped_documents<'a, S, F>(
    mut inner: S,
    map: ChunkMap,
    num_docs: u32,
    budget: Option<SharedThreshold>,
    seek: F,
) -> crate::Result<Box<dyn Scorer + 'a>>
where
    S: Scorer + 'a,
    F: Fn(&mut S, DocId) -> DocId + Send + Sync + 'a,
{
    if map.is_doc_ordered() {
        return Ok(Box::new(inner));
    }
    if num_docs as usize > super::filtered::MAX_FILTER_BITMAP_DOCS {
        return Err(crate::Error::Query(
            "mapped text exceeds the 16 MiB document bitmap budget".into(),
        ));
    }
    let mut matches = DocBitset::new(num_docs);
    let mut visited = 0usize;
    while inner.doc() != TERMINATED {
        if visited.is_multiple_of(1024)
            && budget
                .as_ref()
                .is_some_and(SharedThreshold::stop_if_expired)
        {
            return Ok(Box::new(EmptyScorer));
        }
        matches.set(map.doc_id(inner.doc()));
        inner.advance();
        visited += 1;
    }
    let mut scorer = DocumentMappedScorer {
        inner,
        map,
        matches,
        seek,
        budget,
        current: TERMINATED,
    };
    scorer.position(0);
    Ok(Box::new(scorer))
}

struct DocumentMappedScorer<S, F> {
    inner: S,
    map: ChunkMap,
    matches: DocBitset,
    seek: F,
    budget: Option<SharedThreshold>,
    current: DocId,
}

impl<S: Scorer, F: Fn(&mut S, DocId) -> DocId> DocumentMappedScorer<S, F> {
    fn position(&mut self, target: DocId) {
        self.current = TERMINATED;
        if self
            .budget
            .as_ref()
            .is_some_and(SharedThreshold::stop_if_expired)
        {
            return;
        }
        let Some(doc) = self.matches.next_set_bit(target) else {
            return;
        };
        let (_, slot) = self
            .map
            .slots_for_document(doc)
            .next()
            .expect("validated document map");
        if (self.seek)(&mut self.inner, slot) == slot {
            self.current = doc;
        }
    }
}

impl<S: Scorer, F: Fn(&mut S, DocId) -> DocId + Send + Sync> DocSet for DocumentMappedScorer<S, F> {
    fn doc(&self) -> DocId {
        self.current
    }
    fn advance(&mut self) -> DocId {
        if self.current != TERMINATED {
            self.position(self.current + 1);
        }
        self.current
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if target > self.current {
            self.position(target);
        }
        self.current
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl<S: Scorer, F: Fn(&mut S, DocId) -> DocId + Send + Sync> Scorer for DocumentMappedScorer<S, F> {
    fn score(&self) -> Score {
        self.inner.score()
    }
    fn matched_positions(&self) -> Option<MatchedPositions> {
        self.inner.matched_positions()
    }
}
