//! Build-time diagnostics for sparse vector indexes.
//!
//! Gated behind the `diagnostics` feature flag.

use crate::structures::SparseSkipEntry;

/// Validate serialized block data: count>0, contiguous offsets.
pub(super) fn validate_serialized_blocks(
    dim_id: u32,
    block_data: &[u8],
    skip_entries: &[SparseSkipEntry],
) -> crate::Result<()> {
    for (i, entry) in skip_entries.iter().enumerate() {
        let off = entry.offset as usize;
        if off + 2 > block_data.len() {
            return Err(crate::Error::Corruption(format!(
                "[build] dim_id={} block={}/{}: offset {} + 2 > data_len {}",
                dim_id,
                i,
                skip_entries.len(),
                off,
                block_data.len()
            )));
        }
        let cnt = u16::from_le_bytes([block_data[off], block_data[off + 1]]);
        if cnt == 0 {
            let hex: String = block_data[off..]
                .iter()
                .take(32)
                .map(|x| format!("{x:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            return Err(crate::Error::Corruption(format!(
                "[build] dim_id={} block={}/{}: count=0 (first_32=[{}])",
                dim_id,
                i,
                skip_entries.len(),
                hex
            )));
        }
        if i + 1 < skip_entries.len() {
            let expected = entry.offset + entry.length as u64;
            let actual = skip_entries[i + 1].offset;
            if expected != actual {
                return Err(crate::Error::Corruption(format!(
                    "[build] dim_id={} block={}: non-contiguous: {} + {} = {} != {}",
                    dim_id, i, entry.offset, entry.length, expected, actual
                )));
            }
        }
    }
    Ok(())
}
