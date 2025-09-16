//! Field-local traversal admission and logical predicates at collection boundaries.

use std::sync::Arc;

use crate::segment::{SegmentReader, chunk_map::ChunkMap};

use super::{DocBitset, Query, Scorer, ScorerOptions};

/// A ranked executor compares stable IDs before truncation. Its retained hits
/// re-enter the physical stream without another heap or corpus-sized buffer.
pub(super) fn physical_results(
    hits: &mut [super::ScoredDoc],
    reader: &SegmentReader,
    field: Option<crate::Field>,
) {
    if let Some(map) = field.and_then(|field| reader.chunk_map(field)) {
        for hit in hits {
            hit.doc_id = map
                .document_slot(hit.doc_id)
                .expect("validated document map");
        }
    }
}

pub(super) fn prepare<'a>(
    reader: &'a SegmentReader,
    query: &dyn Query,
    options: &mut ScorerOptions,
    complete: bool,
) -> Option<&'a ChunkMap> {
    let field = query.physical_text_field(reader, complete || options.collect_positions)?;
    let map = reader.chunk_map(field)?;
    if !map.is_document_map() {
        return None;
    }
    options.physical_text_field = Some(field);
    options.complete_text_matches |= complete;
    // These bits use logical IDs. Apply them once outside physical composition.
    options.eligibility = None;
    Some(map)
}

pub(super) fn filtered<'a>(
    scorer: Box<dyn Scorer + 'a>,
    alive: Option<Arc<DocBitset>>,
    map: Option<&'a ChunkMap>,
) -> Box<dyn Scorer + 'a> {
    if let (Some(map), Some(bits)) = (map, alive.as_ref()) {
        let bits = bits.clone();
        Box::new(super::PredicatedScorer::new(
            scorer,
            vec![Box::new(move |slot| bits.contains(map.doc_id(slot)))],
            Vec::new(),
            Vec::new(),
        ))
    } else {
        super::filtered::filtered(scorer, alive)
    }
}
