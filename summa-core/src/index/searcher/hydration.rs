//! Bounded preparation of independent stored-document reads.

use super::Searcher;
use crate::directories::{Directory, FileHandle};
use crate::dsl::Document;
use crate::query::DocAddress;
use crate::segment::PreparedStoreReads;
use crate::{Error, Result};
use rustc_hash::FxHashSet;

/// Largest document window accepted by [`Searcher::prepare_document_reads`].
pub const DOCUMENT_READ_BATCH_SIZE: usize = FileHandle::MAX_BATCH_RANGES;

/// Compressed read-ahead borrowing the searcher that owns its immutable files.
/// Documents are decoded individually, in the caller's chosen order; the batch
/// does not retain decoded documents or bypass the shared store cache.
pub struct DocumentReadBatch<'a, D: Directory + 'static> {
    searcher: &'a Searcher<D>,
    addresses: Vec<DocAddress>,
    reads: Vec<(usize, PreparedStoreReads)>,
}

impl<D: Directory + 'static> Searcher<D> {
    /// Whether any stored-document payload uses explicit asynchronous reads.
    /// Callers may keep ordinary per-document hydration for mapped/RAM readers,
    /// avoiding batch-planning allocations when no reads can be overlapped.
    pub fn has_lazy_document_reads(&self) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.has_lazy_document_reads())
    }

    /// Prepare independent lazy store reads for at most 32 document addresses.
    /// Keeps at most 8 MiB of compressed read-ahead across all segments. Cache
    /// hits, mapped handles, and blocks beyond that allowance use demand access.
    /// No documents are omitted. An oversized window fails before doing I/O.
    pub async fn prepare_document_reads(
        &self,
        addresses: &[DocAddress],
    ) -> Result<DocumentReadBatch<'_, D>> {
        if addresses.len() > DOCUMENT_READ_BATCH_SIZE {
            return Err(Error::Query(
                "document read batch exceeds 32 addresses".into(),
            ));
        }
        let mut groups: Vec<(usize, Vec<u32>)> = Vec::new();
        for address in addresses {
            let Some(segment) = address
                .segment_id_u128()
                .and_then(|id| self.segment_map.get(&id))
                .copied()
            else {
                continue;
            };
            if let Some((_, docs)) = groups.iter_mut().find(|(id, _)| *id == segment) {
                docs.push(address.doc_id);
            } else {
                groups.push((segment, vec![address.doc_id]));
            }
        }
        let mut remaining = FileHandle::MAX_BATCH_BYTES;
        let mut plans = Vec::with_capacity(groups.len());
        for (segment, docs) in groups {
            plans.push((
                segment,
                self.segments[segment].plan_document_reads(&docs, &mut remaining)?,
            ));
        }
        let requests: Vec<_> = plans.iter().flat_map(|(_, plan)| plan.requests()).collect();
        let mut bytes = FileHandle::read_many(&requests).await?.into_iter();
        let reads = plans
            .into_iter()
            .map(|(segment, plan)| (segment, plan.complete(&mut bytes)))
            .collect();
        Ok(DocumentReadBatch {
            searcher: self,
            addresses: addresses.to_vec(),
            reads,
        })
    }
}

impl<D: Directory + 'static> DocumentReadBatch<'_, D> {
    /// Decode one input position, applying the ordinary visibility and field
    /// hydration rules. Duplicate addresses and missing documents are preserved.
    pub async fn get(
        &self,
        position: usize,
        fields: Option<&FxHashSet<u32>>,
    ) -> Result<Option<Document>> {
        let address = self
            .addresses
            .get(position)
            .ok_or_else(|| Error::Query("document batch position out of bounds".into()))?;
        let Some(segment) = address
            .segment_id_u128()
            .and_then(|id| self.searcher.segment_map.get(&id))
            .copied()
        else {
            return Ok(None);
        };
        let reads = &self
            .reads
            .iter()
            .find(|(id, _)| *id == segment)
            .expect("prepared segment")
            .1;
        self.searcher.segments[segment]
            .doc_with_reads(address.doc_id, fields, reads)
            .await
    }
}
