//! POS6 metadata, independent of the packed payload pages.
use super::*;

pub(super) const COMPACT_MAGIC: u32 = 0x3653_4f50;
const GROUP: usize = 8;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid compact position directory",
    )
}

pub(super) fn is_compact(raw: &[u8]) -> bool {
    raw.len() >= FOOTER
        && u32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) == COMPACT_MAGIC
}

pub(super) fn directory_len(blocks: usize) -> Option<usize> {
    blocks
        .div_ceil(GROUP)
        .checked_mul(INDEX_ENTRY)?
        .checked_add(blocks.checked_mul(2)?)
}

pub(super) fn descriptor(header: &[u8]) -> io::Result<u16> {
    if header.len() < BLOCK_HEADER {
        return Err(invalid());
    }
    let count = u16::from_le_bytes(header[..2].try_into().unwrap());
    if !(1..=128).contains(&count) || header[3] > 1 || header[2] > 32 {
        return Err(invalid());
    }
    let tag = (count - 1) | (u16::from(header[2]) << 7) | (u16::from(header[3]) << 13);
    parts(tag)?;
    Ok(tag)
}

pub(super) fn parts(tag: u16) -> io::Result<(usize, u8, u8, usize)> {
    let count = usize::from(tag & 127) + 1;
    let width = ((tag >> 7) & 63) as u8;
    let codec = ((tag >> 13) & 1) as u8;
    if tag >> 14 != 0 {
        return Err(invalid());
    }
    let bytes = match (codec, width) {
        (0, 0 | 8 | 16 | 32) => count * usize::from(width / 8),
        (1, 0..=32) if count == POSITION_STREAM_BLOCK => bitpacking4x::encoded_len(count, width),
        _ => return Err(invalid()),
    };
    Ok((count, width, codec, bytes))
}

#[inline]
pub(super) fn tag(index: &[u8], blocks: usize, block: usize) -> u16 {
    let at = blocks.div_ceil(GROUP) * INDEX_ENTRY + block * 2;
    u16::from_le_bytes(index[at..at + 2].try_into().unwrap())
}

/// Requires writer-produced or admitted metadata. Scans at most seven descriptors.
#[inline]
pub(super) fn entry(index: &[u8], blocks: usize, block: usize) -> (usize, u64) {
    let group = block / GROUP;
    let (mut offset, mut value) = PositionStream::index_entry(index, 0, group);
    for i in group * GROUP..block {
        let descriptor = tag(index, blocks, i);
        let count = usize::from(descriptor & 127) + 1;
        let width = usize::from((descriptor >> 7) & 63);
        offset += (count * width).div_ceil(8);
        value += count as u64;
    }
    (offset, value)
}

/// Search only coarse logical checkpoints, then walk at most one group.
pub(super) fn locate(
    index: &[u8],
    blocks: usize,
    cursor: u64,
    forward: Option<usize>,
) -> Option<(usize, usize)> {
    let groups = blocks.div_ceil(GROUP);
    let mut low = forward.map_or(0, |block| (block / GROUP).min(groups));
    let mut high = groups;
    if forward.is_some() {
        high = low.saturating_add(1).min(groups);
        let mut step = 1usize;
        while high < groups && PositionStream::index_entry(index, 0, high).1 <= cursor {
            low = high;
            step = step.saturating_mul(2);
            high = high.saturating_add(step).min(groups);
        }
    }
    while low < high {
        let middle = low + (high - low) / 2;
        if PositionStream::index_entry(index, 0, middle).1 <= cursor {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let group = low.checked_sub(1)?;
    let mut value = PositionStream::index_entry(index, 0, group).1;
    for block in group * GROUP..((group + 1) * GROUP).min(blocks) {
        let count = u64::from(tag(index, blocks, block) & 127) + 1;
        if cursor < value + count {
            return Some((block, usize::try_from(cursor.checked_sub(value)?).ok()?));
        }
        value += count;
    }
    None
}

pub(super) fn validate(
    index: &[u8],
    blocks: usize,
    payload_len: usize,
    total: u64,
) -> io::Result<()> {
    validate_with(index, blocks, payload_len, total, || Ok(()))
}

pub(super) fn validate_with(
    index: &[u8],
    blocks: usize,
    payload_len: usize,
    total: u64,
    mut check: impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    check()?;
    if directory_len(blocks) != Some(index.len()) {
        return Err(invalid());
    }
    let mut offset = 0usize;
    let mut value = 0u64;
    for block in 0..blocks {
        if block.is_multiple_of(4096) {
            check()?;
        }
        if block.is_multiple_of(GROUP)
            && PositionStream::index_entry(index, 0, block / GROUP) != (offset, value)
        {
            return Err(invalid());
        }
        let (count, _, _, bytes) = parts(tag(index, blocks, block))?;
        offset = offset.checked_add(bytes).ok_or_else(invalid)?;
        value = value.checked_add(count as u64).ok_or_else(invalid)?;
    }
    if offset != payload_len || value != total {
        return Err(invalid());
    }
    Ok(())
}

pub(super) fn write<W: Write>(
    writer: &mut W,
    index: &[(u32, u64)],
    tags: &[u16],
    check: &mut impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    for (i, &(offset, value)) in index.iter().enumerate().step_by(GROUP) {
        if i.is_multiple_of(4096) {
            check()?;
        }
        writer.write_u32::<LittleEndian>(offset)?;
        writer.write_u64::<LittleEndian>(value)?;
    }
    for (i, &tag) in tags.iter().enumerate() {
        if i.is_multiple_of(4096) {
            check()?;
        }
        writer.write_u16::<LittleEndian>(tag)?;
    }
    Ok(())
}
