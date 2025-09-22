//! Copyable document lookups into exact binary ANN codes.
//!
//! Rows contain a document ID, ordinal, and (span, row) address. Merge copies
//! rows verbatim and rebases only block document bases and span byte offsets.
use crate::directories::OwnedBytes;
use std::io;

pub(super) const MAGIC: u32 = 0x314c_5658; // XVL1
pub(super) const VERSION: u32 = 1;
pub(super) const HEADER_SIZE: usize = 32;
pub(super) const ENTRY_SIZE: usize = 14;
const SPAN_SIZE: usize = 12;
const BLOCK_SIZE: usize = 16;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

#[derive(Clone, Debug)]
struct Block {
    first_row: usize,
    #[cfg(feature = "native")]
    count: usize,
    doc_base: u32,
    first_span: usize,
    spans: usize,
}

#[derive(Clone, Debug)]
pub(super) struct VectorLocations {
    spans: OwnedBytes,
    blocks: Vec<Block>,
}

pub(super) struct OpenedLocations {
    pub dim: usize,
    pub count: usize,
    pub ann_len: u64,
    pub rows: OwnedBytes,
    pub layout: VectorLocations,
}

impl VectorLocations {
    pub(super) fn open(bytes: OwnedBytes) -> io::Result<OpenedLocations> {
        if bytes.len() < HEADER_SIZE
            || u32_at(&bytes, 0) != MAGIC
            || u32_at(&bytes, 12) != VERSION
            || u32_at(&bytes, 28) != 0
        {
            return Err(invalid("unsupported or truncated exact-vector lookup"));
        }
        let dim = u32_at(&bytes, 4) as usize;
        let count = u32_at(&bytes, 8) as usize;
        let num_blocks = u32_at(&bytes, 24) as usize;
        if count == 0 || num_blocks == 0 || num_blocks > count {
            return Err(invalid("invalid exact-vector block count"));
        }
        let rows_end = count
            .checked_mul(ENTRY_SIZE)
            .and_then(|n| HEADER_SIZE.checked_add(n))
            .ok_or_else(|| invalid("vector map size overflow"))?;
        let directory_start = num_blocks
            .checked_mul(BLOCK_SIZE)
            .and_then(|n| bytes.len().checked_sub(n))
            .filter(|&at| at >= rows_end)
            .ok_or_else(|| invalid("truncated vector map directory"))?;
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut first_row = 0usize;
        let mut first_span = 0usize;
        for block in bytes[directory_start..].chunks_exact(BLOCK_SIZE) {
            let n = u32_at(block, 0) as usize;
            let spans = u32_at(block, 8) as usize;
            if n == 0 || spans == 0 || u32_at(block, 12) != 0 {
                return Err(invalid("invalid vector map block"));
            }
            blocks.push(Block {
                first_row,
                #[cfg(feature = "native")]
                count: n,
                doc_base: u32_at(block, 4),
                first_span,
                spans,
            });
            first_row = first_row
                .checked_add(n)
                .ok_or_else(|| invalid("vector map row count overflow"))?;
            first_span = first_span
                .checked_add(spans)
                .ok_or_else(|| invalid("vector map span count overflow"))?;
        }
        if first_row != count
            || first_span.checked_mul(SPAN_SIZE) != Some(directory_start - rows_end)
        {
            return Err(invalid("vector map directory does not cover its payload"));
        }
        let ann_len = u64_at(&bytes, 16);
        for span in bytes[rows_end..directory_start].chunks_exact(SPAN_SIZE) {
            let rows = u32_at(span, 8);
            let end = u64::from(rows)
                .checked_mul((dim / 8) as u64)
                .and_then(|size| u64_at(span, 0).checked_add(size));
            if rows == 0 || end.is_none_or(|end| end > ann_len) {
                return Err(invalid("vector lookup span exceeds ANN payload"));
            }
        }
        Ok(OpenedLocations {
            dim,
            count,
            ann_len,
            rows: bytes.slice(HEADER_SIZE..rows_end),
            layout: Self {
                spans: bytes.slice(rows_end..directory_start),
                blocks,
            },
        })
    }

    pub(super) fn spans(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.spans
            .chunks_exact(SPAN_SIZE)
            .map(|span| (u64_at(span, 0), u32_at(span, 8)))
    }

    /// Sequential admission traversal: block bases are selected once per block,
    /// rather than binary-searched for each row.
    pub(super) fn addresses<'a>(
        &'a self,
        rows: &'a [u8],
    ) -> impl Iterator<Item = io::Result<(u32, usize, u32)>> + 'a {
        self.blocks.iter().enumerate().flat_map(move |(i, block)| {
            let end = self
                .blocks
                .get(i + 1)
                .map_or(rows.len(), |next| next.first_row * ENTRY_SIZE);
            rows[block.first_row * ENTRY_SIZE..end]
                .chunks_exact(ENTRY_SIZE)
                .map(move |row| {
                    let address = u64_at(row, 6);
                    let span = (address >> 32) as usize;
                    if span >= block.spans {
                        return Err(invalid("vector lookup span is out of range"));
                    }
                    Ok((block.doc_base, block.first_span + span, address as u32))
                })
        })
    }

    fn block(&self, row: usize) -> &Block {
        let at = self.blocks.partition_point(|block| block.first_row <= row);
        &self.blocks[at - 1]
    }

    pub(super) fn doc_base(&self, row: usize) -> u32 {
        self.block(row).doc_base
    }

    pub(super) fn code_offset(&self, row: usize, address: u64, width: usize) -> io::Result<u64> {
        let block = self.block(row);
        let span = (address >> 32) as usize;
        let in_span = address as u32;
        if span >= block.spans {
            return Err(invalid("vector lookup span is out of range"));
        }
        let at = (block.first_span + span) * SPAN_SIZE;
        let base = u64_at(&self.spans, at);
        if in_span >= u32_at(&self.spans, at + 8) {
            return Err(invalid("vector lookup row is out of range"));
        }
        u64::from(in_span)
            .checked_mul(width as u64)
            .and_then(|delta| base.checked_add(delta))
            .ok_or_else(|| invalid("vector lookup offset overflow"))
    }

    pub(super) fn serialized_len(&self, rows: usize) -> u64 {
        HEADER_SIZE as u64
            + rows as u64 * ENTRY_SIZE as u64
            + self.spans.len() as u64
            + self.blocks.len() as u64 * BLOCK_SIZE as u64
    }

    pub(super) fn heap_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Block>()
    }
}

#[cfg(feature = "native")]
mod writer;

#[cfg(feature = "native")]
pub(crate) use writer::{ExactLocations, write_copied_locations};

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn lookup_directory_rejects_truncation_and_unknown_versions() {
        let mut builder = ExactLocations::default();
        let span = builder.span(100, 2).unwrap();
        builder.push(4, 0, u64::from(span) << 32).unwrap();
        builder.push(7, 3, (u64::from(span) << 32) | 1).unwrap();
        let mut bytes = Vec::new();
        builder.write(24, 2, 200, &mut bytes, None).unwrap();
        for end in 0..bytes.len() {
            assert!(
                VectorLocations::open(OwnedBytes::new(bytes[..end].to_vec())).is_err(),
                "prefix {end}"
            );
        }
        let opened = VectorLocations::open(OwnedBytes::new(bytes.clone())).unwrap();
        assert_eq!(opened.layout.code_offset(1, 1, 3).unwrap(), 103);
        assert!(opened.layout.code_offset(0, 1u64 << 32, 3).is_err());
        assert!(opened.layout.code_offset(0, 2, 3).is_err());
        bytes[12..16].copy_from_slice(&2u32.to_le_bytes());
        assert!(VectorLocations::open(OwnedBytes::new(bytes)).is_err());
    }
}
