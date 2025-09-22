//! Bounded dictionary expansion and canonical posting reads.
use super::{MAX_PREFIX_POSTINGS, MAX_PREFIX_TERMS, SegmentReader, checked_file_range};
use crate::dsl::Field;
use crate::structures::BlockPostingList;
use crate::{Error, Result};

// Inline metadata stays allocation-free; external lists retain borrowed views.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum ExpandedPosting {
    Inline(crate::structures::DecodedInlinePostings),
    External(crate::structures::postings::DeferredPosting),
}

impl ExpandedPosting {
    pub(crate) fn doc_count(&self) -> u32 {
        match self {
            Self::Inline(postings) => postings.docs().len() as u32,
            Self::External(postings) => postings.doc_count(),
        }
    }

    fn into_block(self) -> Result<BlockPostingList> {
        match self {
            Self::External(postings) => Ok(postings.into_list()),
            Self::Inline(postings) => {
                let mut list = crate::structures::PostingList::with_capacity(postings.docs().len());
                for (&doc, &tf) in postings.docs().iter().zip(postings.frequencies()) {
                    list.push(doc, tf);
                }
                Ok(BlockPostingList::from_posting_list(&list)?)
            }
        }
    }
}

impl SegmentReader {
    /// Read the bounded union inputs for a nonempty literal prefix.
    pub async fn get_prefix_postings(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<BlockPostingList>> {
        self.get_prefix_expansion(field, prefix)
            .await?
            .into_iter()
            .map(ExpandedPosting::into_block)
            .collect()
    }

    pub(crate) async fn get_prefix_expansion(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<ExpandedPosting>> {
        if prefix.is_empty() {
            return Err(Error::Query("prefix must not be empty".into()));
        }
        self.get_matching_postings(field, &[prefix.to_vec()], "prefix", usize::MAX, |_| true)
            .await
    }

    pub(crate) async fn get_matching_postings(
        &self,
        field: Field,
        prefixes: &[Vec<u8>],
        label: &str,
        max_scanned: usize,
        mut accepts: impl FnMut(&[u8]) -> bool + Send,
    ) -> Result<Vec<ExpandedPosting>> {
        let mut entries = Vec::new();
        let mut scanned = 0usize;
        let mut truncated = false;
        for prefix in prefixes {
            let mut key_prefix = Vec::with_capacity(4 + prefix.len());
            key_prefix.extend_from_slice(&field.0.to_le_bytes());
            key_prefix.extend_from_slice(prefix);
            let remaining = max_scanned.saturating_sub(scanned);
            let (mut range, more) = self
                .term_dict
                .prefix_scan_values(
                    &key_prefix,
                    MAX_PREFIX_TERMS - entries.len(),
                    remaining,
                    |key| {
                        scanned += 1;
                        accepts(&key[4..])
                    },
                )
                .await?;
            entries.append(&mut range);
            if more {
                truncated = true;
                break;
            }
        }
        if truncated {
            return Err(Error::Query(format!(
                "{label} expands to more than {MAX_PREFIX_TERMS} terms"
            )));
        }
        let posting_count: u64 = entries
            .iter()
            .map(|term_info| term_info.doc_freq() as u64)
            .sum();
        if posting_count > MAX_PREFIX_POSTINGS {
            return Err(Error::Query(format!(
                "{label} expands to {posting_count} postings (maximum {MAX_PREFIX_POSTINGS})"
            )));
        }
        let mut results = Vec::with_capacity(entries.len());
        let postings = self.postings.for_expansion();

        for term_info in entries {
            if term_info.is_inline() {
                let inline = term_info
                    .decode_inline_fixed()
                    .ok_or_else(|| Error::Corruption("invalid expanded inline postings".into()))?;
                results.push(ExpandedPosting::Inline(inline));
            } else if let Some((posting_offset, posting_len)) = term_info.external_info() {
                let range = checked_file_range(
                    posting_offset,
                    posting_len,
                    self.postings.file().len(),
                    "expanded term posting",
                )?;
                results.push(ExpandedPosting::External(
                    postings.read_deferred(range).await?,
                ));
            }
        }

        Ok(results)
    }

    #[cfg(feature = "sync")]
    /// Read the bounded union inputs for a nonempty literal prefix.
    pub fn get_prefix_postings_sync(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<BlockPostingList>> {
        self.get_prefix_expansion_sync(field, prefix)?
            .into_iter()
            .map(ExpandedPosting::into_block)
            .collect()
    }

    #[cfg(feature = "sync")]
    pub(crate) fn get_prefix_expansion_sync(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<ExpandedPosting>> {
        if prefix.is_empty() {
            return Err(Error::Query("prefix must not be empty".into()));
        }
        self.get_matching_postings_sync(field, &[prefix.to_vec()], "prefix", usize::MAX, |_| true)
    }

    #[cfg(feature = "sync")]
    pub(crate) fn get_matching_postings_sync(
        &self,
        field: Field,
        prefixes: &[Vec<u8>],
        label: &str,
        max_scanned: usize,
        mut accepts: impl FnMut(&[u8]) -> bool + Send,
    ) -> Result<Vec<ExpandedPosting>> {
        let mut entries = Vec::new();
        let mut scanned = 0usize;
        let mut truncated = false;
        for prefix in prefixes {
            let mut key_prefix = Vec::with_capacity(4 + prefix.len());
            key_prefix.extend_from_slice(&field.0.to_le_bytes());
            key_prefix.extend_from_slice(prefix);
            let remaining = max_scanned.saturating_sub(scanned);
            let (mut range, more) = self.term_dict.prefix_scan_values_sync(
                &key_prefix,
                MAX_PREFIX_TERMS - entries.len(),
                remaining,
                |key| {
                    scanned += 1;
                    accepts(&key[4..])
                },
            )?;
            entries.append(&mut range);
            if more {
                truncated = true;
                break;
            }
        }
        if truncated {
            return Err(Error::Query(format!(
                "{label} expands to more than {MAX_PREFIX_TERMS} terms"
            )));
        }
        let posting_count: u64 = entries
            .iter()
            .map(|term_info| term_info.doc_freq() as u64)
            .sum();
        if posting_count > MAX_PREFIX_POSTINGS {
            return Err(Error::Query(format!(
                "{label} expands to {posting_count} postings (maximum {MAX_PREFIX_POSTINGS})"
            )));
        }
        let mut results = Vec::with_capacity(entries.len());
        let postings = self.postings.for_expansion();

        for term_info in entries {
            if term_info.is_inline() {
                let inline = term_info
                    .decode_inline_fixed()
                    .ok_or_else(|| Error::Corruption("invalid expanded inline postings".into()))?;
                results.push(ExpandedPosting::Inline(inline));
            } else if let Some((posting_offset, posting_len)) = term_info.external_info() {
                let range = checked_file_range(
                    posting_offset,
                    posting_len,
                    self.postings.file().len(),
                    "expanded term posting",
                )?;
                results.push(ExpandedPosting::External(
                    postings.read_deferred_sync(range)?,
                ));
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
    use crate::{Document, RamDirectory, SchemaBuilder};
    use std::sync::Arc;

    #[tokio::test]
    async fn malformed_inline_expansion_is_an_error_in_both_execution_modes() {
        use crate::directories::{FileHandle, OwnedBytes};
        use crate::structures::{AsyncSSTableReader, SSTableWriter, TermInfo};
        let mut schema = SchemaBuilder::default();
        let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
        let schema = Arc::new(schema.build());
        let directory = RamDirectory::new();
        let mut builder =
            SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
        let mut document = Document::new();
        document.add_text(field, "alpha");
        builder.add_document(document).unwrap();
        let id = SegmentId::new();
        builder.build(&directory, id, None).await.unwrap();
        let mut reader = SegmentReader::open(&directory, id, schema, 16)
            .await
            .unwrap();
        let mut dictionary = SSTableWriter::<_, TermInfo>::new(Vec::new());
        let mut key = field.0.to_le_bytes().to_vec();
        key.extend_from_slice(b"alpha");
        // Valid metadata envelope, truncated posting payload: doc ID but no TF.
        dictionary
            .insert(
                &key,
                &TermInfo::Inline {
                    doc_freq: 1,
                    data: [0; 16],
                    data_len: 1,
                },
            )
            .unwrap();
        reader.term_dict = Arc::new(
            AsyncSSTableReader::open(
                FileHandle::from_bytes(OwnedBytes::new(dictionary.finish().unwrap())),
                8,
            )
            .await
            .unwrap(),
        );
        assert!(
            reader
                .get_prefix_expansion(field, b"a")
                .await
                .unwrap_err()
                .to_string()
                .contains("invalid expanded inline postings")
        );
        #[cfg(feature = "sync")]
        assert!(
            reader
                .get_prefix_expansion_sync(field, b"a")
                .unwrap_err()
                .to_string()
                .contains("invalid expanded inline postings")
        );
    }

    #[tokio::test]
    async fn disjoint_dictionary_ranges_share_one_scan_and_expansion_budget() {
        let mut schema = SchemaBuilder::default();
        let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
        let schema = Arc::new(schema.build());
        let directory = RamDirectory::new();
        let mut builder =
            SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
        let mut document = Document::new();
        for prefix in ["alpha", "beta"] {
            for n in 0..600 {
                document.add_text(field, format!("{prefix}{n:03}"));
            }
        }
        builder.add_document(document).unwrap();
        let id = SegmentId::new();
        builder.build(&directory, id, None).await.unwrap();
        let reader = SegmentReader::open(&directory, id, schema, 16)
            .await
            .unwrap();
        let prefixes = vec![b"alpha".to_vec(), b"beta".to_vec()];
        for (scan_budget, accepts, expected) in [
            (1200, false, "ok"),
            (1199, false, "scan exceeds"),
            (1200, true, "more than 1024"),
        ] {
            let outcome = reader
                .get_matching_postings(field, &prefixes, "regex", scan_budget, |_| accepts)
                .await;
            if expected == "ok" {
                assert!(outcome.unwrap().is_empty());
            } else {
                assert!(outcome.unwrap_err().to_string().contains(expected));
            }
            #[cfg(feature = "sync")]
            {
                let outcome = reader.get_matching_postings_sync(
                    field,
                    &prefixes,
                    "regex",
                    scan_budget,
                    |_| accepts,
                );
                if expected == "ok" {
                    assert!(outcome.unwrap().is_empty());
                } else {
                    assert!(outcome.unwrap_err().to_string().contains(expected));
                }
            }
        }
    }
}
