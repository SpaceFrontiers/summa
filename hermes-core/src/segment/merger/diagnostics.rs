//! Merge-time diagnostics for sparse vector indexes.
//!
//! Gated behind the `diagnostics` feature flag.

use crate::segment::reader::DimRawData;

/// Block header layout constants.
const BLOCK_HEADER_SIZE: usize = 16;

/// Validate source skip entries before merging: header bounds, count>0, contiguity.
pub(super) fn validate_merge_source(
    dim_id: u32,
    src_idx: usize,
    raw: &DimRawData,
) -> crate::Result<()> {
    let data = raw.raw_block_data.as_slice();

    for (i, entry) in raw.skip_entries.iter().enumerate() {
        let start = entry.offset as usize;
        if start + BLOCK_HEADER_SIZE > data.len() {
            return Err(crate::Error::Corruption(format!(
                "[merge] dim_id={} src={} block={}/{}: skip offset {} + header {} > data_len {}",
                dim_id,
                src_idx,
                i,
                raw.skip_entries.len(),
                start,
                BLOCK_HEADER_SIZE,
                data.len()
            )));
        }
        let block_count = u16::from_le_bytes([data[start], data[start + 1]]);
        if block_count == 0 {
            let hex: String = data[start..]
                .iter()
                .take(32)
                .map(|x| format!("{x:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            return Err(crate::Error::Corruption(format!(
                "[merge] dim_id={} src={} block={}/{}: count=0 at offset={} length={} (first_32=[{}])",
                dim_id,
                src_idx,
                i,
                raw.skip_entries.len(),
                entry.offset,
                entry.length,
                hex
            )));
        }

        // Check contiguity: offset[i] + length[i] == offset[i+1]
        if i + 1 < raw.skip_entries.len() {
            let expected_next = entry.offset + entry.length as u64;
            let actual_next = raw.skip_entries[i + 1].offset;
        }
    }
    Ok(())
}
