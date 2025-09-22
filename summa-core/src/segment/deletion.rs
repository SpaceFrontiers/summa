//! Immutable row visibility. Files are owned by the ordinary segment tracker.
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::directories::Directory;
use crate::query::DocBitset;
use crate::{Error, Result};

/// A compact metadata reference to one immutable visibility generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletionMeta {
    /// Independently tracked file identity, not the data segment identity.
    pub id: String,
    pub num_deleted: u32,
}

impl DeletionMeta {
    pub(crate) fn path(&self) -> Result<PathBuf> {
        let id = super::SegmentId::from_hex(&self.id)
            .ok_or_else(|| Error::Corruption("invalid deletion file identity".into()))?;
        Ok(super::SegmentFiles::new(id.0).deletions)
    }

    pub(crate) async fn load<D: Directory>(
        &self,
        directory: &D,
        num_docs: u32,
    ) -> Result<Arc<DocBitset>> {
        if self.num_deleted == 0 || self.num_deleted > num_docs {
            return Err(Error::Corruption("invalid deletion count".into()));
        }
        let handle = directory.open_read(&self.path()?).await.map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::Corruption(format!("referenced deletion file {} is missing", self.id))
            } else {
                Error::Io(error)
            }
        })?;
        let expected = 24 + (num_docs as u64).div_ceil(64) * 8;
        if handle.len() != expected {
            return Err(Error::Corruption("invalid deletion file length".into()));
        }
        let bytes = handle.read_bytes().await?;
        decode(bytes.as_slice(), num_docs, self.num_deleted).map(Arc::new)
    }
}

// FNV-1a over the canonical little-endian header and payload. This detects
// accidental corruption, including changes that preserve the population count.
fn checksum(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn decode(bytes: &[u8], num_docs: u32, num_deleted: u32) -> Result<DocBitset> {
    let invalid = || Error::Corruption("invalid row deletion bitset".into());
    let expected = 24 + (num_docs as usize).div_ceil(64) * 8;
    if num_deleted > num_docs || bytes.len() != expected || &bytes[..8] != b"HDEL\x01\0\0\0" {
        return Err(invalid());
    }
    let read_u32 = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    if read_u32(8) != num_docs || read_u32(12) != num_deleted {
        return Err(invalid());
    }
    let end = bytes.len() - 8;
    if checksum(bytes[..end].iter().copied())
        != u64::from_le_bytes(bytes[end..].try_into().unwrap())
    {
        return Err(invalid());
    }
    let bits = bytes[16..end]
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect();
    let alive = DocBitset { bits };
    if alive.count() != num_docs - num_deleted || alive.next_set_bit(num_docs).is_some() {
        return Err(invalid());
    }
    Ok(alive)
}

/// Writes a claimed file without a second bitmap-sized serialization buffer.
#[cfg(any(feature = "native", feature = "wasm"))]
pub(crate) async fn write<D: crate::directories::DirectoryWriter>(
    directory: &D,
    id: super::SegmentId,
    num_docs: u32,
    alive: &DocBitset,
) -> Result<DeletionMeta> {
    use std::io::Write;
    let num_deleted = num_docs
        .checked_sub(alive.count())
        .ok_or_else(|| Error::Corruption("deletion population exceeds row count".into()))?;
    if num_deleted == 0
        || alive.bits.len() != (num_docs as usize).div_ceil(64)
        || alive.next_set_bit(num_docs).is_some()
    {
        return Err(Error::Corruption("invalid deletion output".into()));
    }
    let metadata = DeletionMeta {
        id: id.to_hex(),
        num_deleted,
    };
    let mut header = Vec::with_capacity(16);
    header.extend_from_slice(b"HDEL\x01\0\0\0");
    header.extend_from_slice(&num_docs.to_le_bytes());
    header.extend_from_slice(&num_deleted.to_le_bytes());
    let digest = checksum(
        header
            .iter()
            .copied()
            .chain(alive.bits.iter().flat_map(|w| w.to_le_bytes())),
    );
    let mut writer = directory.streaming_writer_cold(&metadata.path()?).await?;
    writer.write_all(&header)?;
    for word in &alive.bits {
        writer.write_all(&word.to_le_bytes())?;
    }
    writer.write_all(&digest.to_le_bytes())?;
    writer.finish()?;
    Ok(metadata)
}

/// Copy deletion bits into a concatenated physical row space, including tails
/// whose source boundary is not aligned to a bitmap word.
#[cfg(feature = "native")]
pub(crate) fn append_dead_rows(
    target: &mut DocBitset,
    source: &DocBitset,
    num_docs: u32,
    offset: u32,
) -> Result<()> {
    if (offset as u64 + num_docs as u64).div_ceil(64) > target.bits.len() as u64 {
        return Err(Error::Corruption(
            "deletion remap exceeds output row space".into(),
        ));
    }
    for (word_index, word) in source.bits.iter().enumerate() {
        let valid = (num_docs as usize - word_index * 64).min(64);
        let padding = if valid == 64 {
            u64::MAX
        } else {
            (1u64 << valid) - 1
        };
        let dead = !word & padding;
        let start = offset as usize + word_index * 64;
        let out = start / 64;
        let shift = start % 64;
        target.bits[out] &= !(dead << shift);
        if shift != 0 && out + 1 < target.bits.len() {
            target.bits[out + 1] &= !(dead >> (64 - shift));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn merging_deletion_masks_preserves_every_unaligned_boundary_and_neighbor() {
        for offset in 0..128 {
            for len in 0..131 {
                let total = offset + len + 65;
                let mut target = DocBitset::all(total);
                // A previously appended source may already have tombstones.
                if offset > 0 {
                    target.clear(offset - 1);
                }
                let mut source = DocBitset::all(len);
                for doc in 0..len {
                    if (doc + len) % 3 != 0 {
                        source.clear(doc);
                    }
                }
                append_dead_rows(&mut target, &source, len, offset).unwrap();
                for doc in 0..total {
                    let expected = if doc >= offset && doc < offset + len {
                        source.contains(doc - offset)
                    } else {
                        offset == 0 || doc != offset - 1
                    };
                    assert_eq!(
                        target.contains(doc),
                        expected,
                        "offset={offset} len={len} row={doc}"
                    );
                }
                assert_eq!(target.next_set_bit(total), None);
            }
        }
    }

    #[tokio::test]
    async fn deletion_files_reject_corruption_and_keep_tail_rows() {
        let dir = crate::directories::RamDirectory::new();
        let mut alive = DocBitset::all(65);
        alive.clear(0);
        let meta = write(&dir, super::super::SegmentId::new(), 65, &alive)
            .await
            .unwrap();
        let loaded = meta.load(&dir, 65).await.unwrap();
        assert!(!loaded.contains(0));
        assert!(loaded.contains(64));
        let bytes = dir
            .open_read(&meta.path().unwrap())
            .await
            .unwrap()
            .read_bytes()
            .await
            .unwrap();
        let mut corrupt = bytes.as_slice().to_vec();
        corrupt[16] ^= 3;
        assert!(decode(&corrupt, 65, 1).is_err());
        assert!(decode(bytes.as_slice(), 64, 1).is_err());
    }
}

/// Bounded key resolution over the fast column's global dictionary ordinals.
#[cfg(any(feature = "native", feature = "wasm"))]
pub(crate) fn target_ordinals<'a>(
    column: &crate::structures::fast_field::FastFieldReader,
    num_docs: u32,
    keys: impl Iterator<Item = &'a str>,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<rustc_hash::FxHashSet<u64>> {
    if column.num_docs != num_docs
        || column.multi
        || column.column_type != crate::structures::fast_field::FastFieldColumnType::TextOrdinal
    {
        return Err(Error::Corruption(
            "primary-key column must contain one text value per physical row".into(),
        ));
    }
    let mut ordinals = rustc_hash::FxHashSet::default();
    for (i, key) in keys.enumerate() {
        if i.is_multiple_of(4096) {
            check_cancelled()?;
        }
        if let Some(ordinal) = column.text_ordinal(key) {
            ordinals.insert(ordinal);
        }
    }
    Ok(ordinals)
}

/// Streaming batch decode; shared by native worker-pool and inline WASM commits.
#[cfg(any(feature = "native", feature = "wasm"))]
pub(crate) fn clear_target_rows(
    column: &crate::structures::fast_field::FastFieldReader,
    ordinals: &rustc_hash::FxHashSet<u64>,
    alive: &mut DocBitset,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<bool> {
    let mut changed = false;
    column.try_scan_single_values(|doc, ordinal| {
        if doc.is_multiple_of(4096) {
            check_cancelled()?;
        }
        if alive.contains(doc) && ordinals.contains(&ordinal) {
            alive.clear(doc);
            changed = true;
        }
        Ok::<_, Error>(())
    })?;
    Ok(changed)
}
