//! Explicit, bounded binary nomination policy. Zero preserves existing defaults.

use crate::dsl::{BinaryDenseVectorConfig, BinaryIndexType};
use crate::segment::VectorSearchResult;

/// Query controls independent of returned document depth and score aggregation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BinaryRetrievalOptions {
    /// IVF/ScaNN probes; zero uses the schema's probe count.
    pub nprobe: usize,
    /// Prefix shortlist size; zero uses the schema's size. Requires prefix IVF.
    pub prefix_rerank: usize,
    /// Highest-scoring ordinals to nominate per document; zero keeps all.
    /// Does not change the document score or score-only backfill semantics.
    pub max_ordinals_per_doc: usize,
}

impl BinaryRetrievalOptions {
    /// Validate before admitting a request or allocating query scratch.
    pub fn validate(self) -> Result<(), String> {
        for (name, value, maximum) in [
            ("nprobe", self.nprobe, 65_536),
            ("prefix_rerank", self.prefix_rerank, 65_536),
            ("max_ordinals_per_doc", self.max_ordinals_per_doc, 64),
        ] {
            if value > maximum {
                return Err(format!("binary {name} exceeds maximum {maximum}"));
            }
        }
        Ok(())
    }

    /// Reject controls a field cannot honor instead of silently ignoring them.
    pub fn validate_field(self, config: &BinaryDenseVectorConfig) -> Result<(), String> {
        self.validate()?;
        if self.nprobe != 0 && config.index_type == BinaryIndexType::Flat {
            return Err("binary nprobe requires an IVF or ScaNN field".into());
        }
        if self.prefix_rerank != 0
            && (config.index_type != BinaryIndexType::Ivf || config.prefix_bits.is_none())
        {
            return Err("binary prefix_rerank requires an IVF field with prefix_bits".into());
        }
        Ok(())
    }

    pub(crate) fn effective_nprobe(self, config: &BinaryDenseVectorConfig) -> usize {
        if self.nprobe == 0 {
            config.nprobe
        } else {
            self.nprobe
        }
    }

    pub(crate) fn effective_prefix_rerank(self, config: &BinaryDenseVectorConfig) -> usize {
        if self.prefix_rerank == 0 {
            config.effective_prefix_rerank()
        } else {
            self.prefix_rerank
        }
    }

    pub(super) fn select_ordinals(self, results: &mut [VectorSearchResult]) {
        let limit = self.max_ordinals_per_doc;
        if limit == 0 {
            return;
        }
        for result in results {
            if result.ordinals.len() > limit {
                // Linear selection, bounded retained output. Keep original
                // document scores; downstream fusion sees only these nominees.
                result.ordinals.select_nth_unstable_by(limit, |a, b| {
                    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                });
                result.ordinals.truncate(limit);
                result.ordinals.sort_unstable_by_key(|entry| entry.0);
            }
        }
    }
}
