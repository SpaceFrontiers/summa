//! Bounded logical probes of MaxScore sparse postings. Skip ranges select only
//! blocks intersecting the candidate documents; no complete posting list loads.
use super::SparseIndex;
use crate::{Error, Result};

#[derive(Default)]
pub(crate) struct SparseProbeBudget {
    probes: usize,
    bytes: u64,
}
impl SparseProbeBudget {
    fn probe(&mut self) -> Result<()> {
        self.probes += 1;
        if self.probes > 2_000_000 {
            return Err(Error::Query(
                "L1 sparse posting probe budget exceeded".into(),
            ));
        }
        Ok(())
    }
    fn read(&mut self, bytes: u32) -> Result<()> {
        self.bytes += u64::from(bytes);
        if self.bytes > 256 * 1024 * 1024 {
            return Err(Error::Query(
                "L1 sparse posting read budget exceeded (256 MiB)".into(),
            ));
        }
        Ok(())
    }
}

impl SparseIndex {
    /// `None` visits all retained dimensions to establish field presence and
    /// stored ordinals. A weighted dimension probes scores using the owning
    /// block decoder, including its quantization-specific multiplication order.
    pub(crate) async fn probe_candidates(
        &self,
        documents: &[u32],
        dimension: Option<(u32, f32)>,
        budget: &mut SparseProbeBudget,
        mut visit: impl FnMut(u32, u16, f32) -> Result<()>,
    ) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        if !documents.windows(2).all(|p| p[0] < p[1])
            || documents.last().is_some_and(|&d| d >= self.total_docs)
        {
            return Err(Error::Query(
                "sparse candidate documents must be sorted, unique and valid".into(),
            ));
        }
        let dimensions = match dimension {
            Some((dim, _)) => match self.dims.find(dim) {
                Some(i) => i..i + 1,
                None => return Ok(()),
            },
            None => 0..self.dims.dim_ids.len(),
        };
        let mut ids = Vec::new();
        let mut ordinals = Vec::new();
        let mut weights = Vec::new();
        for dim in dimensions {
            let mut candidate = 0usize;
            let mut next_block = 0usize;
            let skip_start = self.dims.skip_starts[dim] as usize;
            let blocks = self.dims.skip_counts[dim] as usize;
            while candidate < documents.len() && next_block < blocks {
                budget.probe()?;
                let doc = documents[candidate];
                let mut low = next_block;
                let mut high = blocks;
                while low < high {
                    let mid = low + (high - low) / 2;
                    if self.read_skip_entry(skip_start + mid).last_doc < doc {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                if low == blocks {
                    break;
                }
                let skip = self.read_skip_entry(skip_start + low);
                if skip.first_doc > skip.last_doc {
                    return Err(Error::Corruption("sparse skip range is inverted".into()));
                }
                while candidate < documents.len() && documents[candidate] < skip.first_doc {
                    candidate += 1;
                }
                if candidate == documents.len() {
                    break;
                }
                next_block = low + 1;
                if documents[candidate] > skip.last_doc {
                    continue;
                }
                budget.read(skip.length)?;
                let block = self.load_block_at(dim, low).await?;
                block.decode_doc_ids_into(&mut ids);
                block.decode_ordinals_into(&mut ordinals);
                if let Some((_, weight)) = dimension {
                    block.decode_scored_weights_into(weight, &mut weights);
                }
                if ids.len() != ordinals.len()
                    || dimension.is_some() && weights.len() != ids.len()
                    || ids.first().copied() != Some(skip.first_doc)
                    || ids.last().copied() != Some(skip.last_doc)
                    || !ids.is_sorted()
                {
                    return Err(Error::Corruption(
                        "sparse candidate block disagrees with skip metadata".into(),
                    ));
                }
                let mut selected = candidate;
                for (i, (&doc, &ordinal)) in ids.iter().zip(&ordinals).enumerate() {
                    while selected < documents.len() && documents[selected] < doc {
                        selected += 1;
                    }
                    if documents.get(selected) == Some(&doc) {
                        let value = if dimension.is_some() { weights[i] } else { 0.0 };
                        if !value.is_finite() {
                            return Err(Error::Corruption(
                                "non-finite sparse candidate impact".into(),
                            ));
                        }
                        visit(doc, ordinal, value)?;
                    }
                }
                // Keep a boundary document: its ordinals can span blocks.
                while candidate < documents.len() && documents[candidate] < skip.last_doc {
                    candidate += 1;
                }
            }
        }
        Ok(())
    }
}
