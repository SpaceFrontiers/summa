//! Structural checks before exposing immutable bytes to infallible decoders.
use super::*;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Metadata-only admission for compact fixed-width blocks.
pub(super) fn validate_descriptor(
    header: &[u8],
    first: u32,
    last: u32,
    payload_bytes: usize,
) -> io::Result<usize> {
    let count = u16::from_le_bytes(header[..2].try_into().unwrap()) as usize;
    if first > last
        || last == TERMINATED
        || count == 0
        || count > BLOCK_SIZE
        || count as u64 > u64::from(last) - u64::from(first) + 1
    {
        return Err(invalid("invalid compact posting count or range"));
    }
    let (codec, width) = PostingCodec::from_header_byte(header[2])?;
    let tf_width = header[3];
    if codec == PostingCodec::Pfor
        || tf_width > 32
        || (codec == PostingCodec::Rounded
            && (!matches!(width, 0 | 8 | 16 | 32) || !matches!(tf_width, 0 | 8 | 16 | 32)))
    {
        return Err(invalid("invalid compact posting codec"));
    }
    let gaps = if codec == PostingCodec::Simd4x && count == BLOCK_SIZE {
        count
    } else {
        count - 1
    };
    if packed_bytes(gaps, width) + packed_bytes(count, tf_width) != payload_bytes {
        return Err(invalid(
            "compact posting payload length disagrees with descriptor",
        ));
    }
    Ok(count)
}

fn payload_len(input: &[u8], codec: PostingCodec, count: usize, bits: u8) -> io::Result<usize> {
    if bits > 32 {
        return Err(invalid("invalid posting width"));
    }
    let len = match codec {
        PostingCodec::Rounded => {
            if !matches!(bits, 0 | 8 | 16 | 32) {
                return Err(invalid("invalid rounded posting width"));
            }
            count * (bits as usize / 8)
        }
        PostingCodec::Packed | PostingCodec::Simd4x => packed_bytes(count, bits),
        PostingCodec::Pfor => pfor_payload_len(input, count, bits)?,
    };
    if len > input.len() {
        return Err(invalid("truncated posting payload"));
    }
    if codec == PostingCodec::Pfor {
        let exceptions = input[0] as usize;
        if exceptions > count || (bits == 32 && exceptions != 0) {
            return Err(invalid("invalid posting exception count"));
        }
        let mut previous = None;
        for entry in input[1 + packed_bytes(count, bits)..len].chunks_exact(5) {
            let position = entry[0] as usize;
            let high = u32::from_le_bytes(entry[1..].try_into().unwrap());
            if position >= count
                || previous.is_some_and(|old| position <= old)
                || high == 0
                || high > (u32::MAX >> bits)
            {
                return Err(invalid("invalid posting exception"));
            }
            previous = Some(position);
        }
    }
    Ok(len)
}

/// Returns the bounded posting count; no posting-sized allocation or decode.
pub(super) fn validate_block(stream: &[u8], first: u32, last: u32) -> io::Result<usize> {
    if stream.len() < 8 {
        return Err(invalid("truncated posting block"));
    }
    let count = u16::from_le_bytes(stream[..2].try_into().unwrap()) as usize;
    if first > last
        || last == TERMINATED
        || count == 0
        || count > BLOCK_SIZE
        || count as u64 > u64::from(last) - u64::from(first) + 1
        || u32::from_le_bytes(stream[2..6].try_into().unwrap()) != first
    {
        return Err(invalid("posting block header disagrees with directory"));
    }
    let (codec, bits) = PostingCodec::from_header_byte(stream[6])?;
    // Validate widths even when a singleton has no delta payload.
    if codec == PostingCodec::Rounded && !matches!(bits, 0 | 8 | 16 | 32) {
        return Err(invalid("invalid rounded posting width"));
    }
    let gaps = if codec == PostingCodec::Simd4x && count == BLOCK_SIZE {
        let bytes = payload_len(&stream[8..], codec, count, bits)?;
        if !bitpacking4x::first_gap_is_zero(&stream[8..8 + bytes], bits) {
            return Err(invalid("SIMD posting first gap must be zero"));
        }
        bytes
    } else if count > 1 {
        payload_len(&stream[8..], codec, count - 1, bits)?
    } else {
        0
    };
    let tfs = payload_len(&stream[8 + gaps..], codec, count, stream[7])?;
    if 8 + gaps + tfs != stream.len() {
        return Err(invalid("invalid posting payload length"));
    }
    Ok(count)
}

/// Validate remapped document ranges without decoding or rewriting payloads.
/// `previous_last` spans source boundaries as well as blocks within a source.
pub(super) fn validate_remap(
    l0: &[u8],
    blocks: usize,
    offset: u32,
    previous_last: &mut Option<u32>,
) -> io::Result<()> {
    for i in 0..blocks {
        let (first, last, _, _) = read_l0(l0, i);
        let first = first
            .checked_add(offset)
            .ok_or_else(|| invalid("posting document remapping overflow"))?;
        let last = last
            .checked_add(offset)
            .ok_or_else(|| invalid("posting document remapping overflow"))?;
        if first > last || last == TERMINATED || previous_last.is_some_and(|old| first <= old) {
            return Err(invalid("invalid remapped posting document order"));
        }
        *previous_last = Some(last);
    }
    Ok(())
}

pub(super) fn validate_list(raw: &[u8], footer: &Footer) -> io::Result<()> {
    if footer.l1_count != footer.l0_count.div_ceil(L1_INTERVAL)
        || (footer.l0_count == 0) != (footer.doc_count == 0)
        || (!footer.compact_headers && (footer.l0_count == 0) != (footer.stream_len == 0))
        || u64::from(footer.doc_count) < footer.l0_count as u64
        || u64::from(footer.doc_count) > footer.l0_count as u64 * BLOCK_SIZE as u64
    {
        return Err(invalid("invalid posting list counts"));
    }
    let l0 = &raw[footer.l0_start()..footer.l0_end()];
    let mut previous_last = None;
    let mut total = 0u64;
    let mut previous_cursor = 0;
    for i in 0..footer.l0_count {
        let (first, last, offset, _) = read_l0(l0, i);
        let end = if i + 1 == footer.l0_count {
            footer.stream_len
        } else {
            read_l0(l0, i + 1).2 as usize
        };
        if (i == 0 && offset != 0)
            || previous_last.is_some_and(|old| first <= old)
            || end < offset as usize
            || (!footer.compact_headers && end == offset as usize)
            || end > footer.stream_len
        {
            return Err(invalid("invalid posting directory"));
        }
        total += if footer.compact_headers {
            let at = footer.l0_end() + i * 4;
            validate_descriptor(&raw[at..at + 4], first, last, end - offset as usize)?
        } else {
            validate_block(&raw[offset as usize..end], first, last)?
        } as u64;
        if (i + 1).is_multiple_of(L1_INTERVAL) || i + 1 == footer.l0_count {
            let at = footer.l1_start() + (i / L1_INTERVAL) * L1_SIZE;
            if u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) != last {
                return Err(invalid("posting L1 directory disagrees with L0"));
            }
        }
        if footer.has_cursors {
            let size = footer.cursor_size();
            let at = footer.l1_bounds_end() + i * size;
            let cursor = if footer.short_cursors {
                u64::from(u32::from_le_bytes(raw[at..at + size].try_into().unwrap()))
            } else {
                u64::from_le_bytes(raw[at..at + size].try_into().unwrap())
            };
            if (i == 0 && cursor != 0)
                || cursor < previous_cursor
                || cursor > footer.total_positions
            {
                return Err(invalid("invalid posting position cursor"));
            }
            previous_cursor = cursor;
        }
        previous_last = Some(last);
    }
    if total != u64::from(footer.doc_count) {
        return Err(invalid("posting block counts disagree with footer"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_headers_and_directories_are_rejected_before_iteration() {
        let mut postings = PostingList::new();
        for i in 0..257 {
            postings.push(i * 2, i % 11 + 1);
        }
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let list = BlockPostingList::from_posting_list_with_codec(&postings, codec).unwrap();
            let mut original = Vec::new();
            list.serialize(&mut original).unwrap();
            let footer = Footer::parse(&original).unwrap();
            for (at, value) in [
                (0, 0),                               // zero first block count
                (1, 1),                               // count exceeds block capacity
                (2, 1),                               // first document disagrees with directory
                (6, 0xe1),                            // unsupported codec width
                (7, 33),                              // illegal frequency width
                (footer.l0_start() + 8, 1),           // first offset must be zero
                (footer.l0_start() + L0_SIZE + 8, 0), // overlapping blocks
                (footer.l0_end(), 1),                 // inconsistent L1 last document
            ] {
                let mut corrupt = original.clone();
                corrupt[at] = value;
                assert!(
                    BlockPostingList::deserialize(&corrupt).is_err(),
                    "codec={codec}, byte={at}"
                );
            }
            let reopened = BlockPostingList::deserialize(&original).unwrap();
            let mut encoded = Vec::new();
            reopened.serialize(&mut encoded).unwrap();
            assert_eq!(encoded, original);
            assert_eq!(reopened.iterator().doc(), 0);
        }
    }

    #[test]
    fn malformed_pfor_exception_positions_and_values_are_rejected() {
        // Two documents, one zero-width delta exception; two one-bit TFs.
        let valid = [2, 0, 0, 0, 0, 0, 0x80, 1, 1, 0, 1, 0, 0, 0, 0, 3];
        assert_eq!(validate_block(&valid, 0, 1).unwrap(), 2);
        for (at, value) in [(8, 2), (9, 1), (10, 0), (7, 33)] {
            let mut corrupt = valid;
            corrupt[at] = value;
            assert!(validate_block(&corrupt, 0, 1).is_err(), "byte={at}");
        }
        for end in 0..valid.len() {
            assert!(validate_block(&valid[..end], 0, 1).is_err());
        }
    }
}
