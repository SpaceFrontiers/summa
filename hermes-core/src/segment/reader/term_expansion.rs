//! Bounded dictionary expansion and canonical posting reads.
use super::{MAX_PREFIX_POSTINGS, MAX_PREFIX_TERMS, SegmentReader, checked_file_range};
use crate::dsl::Field;
use crate::structures::BlockPostingList;
use crate::{Error, Result};

impl SegmentReader {
    /// Read the bounded union inputs for a nonempty literal prefix.
    pub async fn get_prefix_postings(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<BlockPostingList>> {
        if prefix.is_empty() {
            return Err(Error::Query("prefix must not be empty".into()));
        }
        self.get_matching_postings(field, prefix, "prefix", usize::MAX, |_| true)
            .await
    }

    pub(crate) async fn get_matching_postings(
        &self,
        field: Field,
        prefix: &[u8],
        label: &str,
        max_scanned: usize,
        mut accepts: impl FnMut(&[u8]) -> bool + Send,
    ) -> Result<Vec<BlockPostingList>> {
        // Build composite key prefix: field_id ++ prefix
        let mut key_prefix = Vec::with_capacity(4 + prefix.len());
        key_prefix.extend_from_slice(&field.0.to_le_bytes());
        key_prefix.extend_from_slice(prefix);

        let (entries, truncated) = self
            .term_dict
            .prefix_scan_filtered(&key_prefix, MAX_PREFIX_TERMS, max_scanned, |key| {
                accepts(&key[4..])
            })
            .await?;
        if truncated {
            return Err(Error::Query(format!(
                "{label} expands to more than {MAX_PREFIX_TERMS} terms"
            )));
        }
        let posting_count: u64 = entries
            .iter()
            .map(|(_, term_info)| term_info.doc_freq() as u64)
            .sum();
        if posting_count > MAX_PREFIX_POSTINGS {
            return Err(Error::Query(format!(
                "{label} expands to {posting_count} postings (maximum {MAX_PREFIX_POSTINGS})"
            )));
        }
        let mut results = Vec::with_capacity(entries.len());

        for (_key, term_info) in entries {
            if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
                let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
                for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs) {
                    posting_list.push(doc_id, tf);
                }
                results.push(BlockPostingList::from_posting_list(&posting_list)?);
            } else if let Some((posting_offset, posting_len)) = term_info.external_info() {
                let range = checked_file_range(
                    posting_offset,
                    posting_len,
                    self.postings.file().len(),
                    "expanded term posting",
                )?;
                results.push(self.postings.read(range).await?);
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
        if prefix.is_empty() {
            return Err(Error::Query("prefix must not be empty".into()));
        }
        self.get_matching_postings_sync(field, prefix, "prefix", usize::MAX, |_| true)
    }

    #[cfg(feature = "sync")]
    pub(crate) fn get_matching_postings_sync(
        &self,
        field: Field,
        prefix: &[u8],
        label: &str,
        max_scanned: usize,
        mut accepts: impl FnMut(&[u8]) -> bool + Send,
    ) -> Result<Vec<BlockPostingList>> {
        let mut key_prefix = Vec::with_capacity(4 + prefix.len());
        key_prefix.extend_from_slice(&field.0.to_le_bytes());
        key_prefix.extend_from_slice(prefix);

        let (entries, truncated) = self.term_dict.prefix_scan_filtered_sync(
            &key_prefix,
            MAX_PREFIX_TERMS,
            max_scanned,
            |key| accepts(&key[4..]),
        )?;
        if truncated {
            return Err(Error::Query(format!(
                "{label} expands to more than {MAX_PREFIX_TERMS} terms"
            )));
        }
        let posting_count: u64 = entries
            .iter()
            .map(|(_, term_info)| term_info.doc_freq() as u64)
            .sum();
        if posting_count > MAX_PREFIX_POSTINGS {
            return Err(Error::Query(format!(
                "{label} expands to {posting_count} postings (maximum {MAX_PREFIX_POSTINGS})"
            )));
        }
        let mut results = Vec::with_capacity(entries.len());

        for (_key, term_info) in entries {
            if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
                let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
                for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs) {
                    posting_list.push(doc_id, tf);
                }
                results.push(BlockPostingList::from_posting_list(&posting_list)?);
            } else if let Some((posting_offset, posting_len)) = term_info.external_info() {
                let range = checked_file_range(
                    posting_offset,
                    posting_len,
                    self.postings.file().len(),
                    "expanded term posting",
                )?;
                results.push(self.postings.read_sync(range)?);
            }
        }

        Ok(results)
    }
}
