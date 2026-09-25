use super::*;

fn encoded(codec: PostingCodec) -> Vec<u8> {
    let mut postings = PostingList::new();
    for doc in 0..257 {
        postings.push(doc * 2, doc % 7 + 1);
    }
    let list = BlockPostingList::from_posting_list_with_ratio_bounds(
        &postings,
        true,
        Some(&|doc| doc % 101 + 8),
        codec,
    )
    .unwrap();
    let mut bytes = Vec::new();
    list.serialize(&mut bytes).unwrap();
    bytes
}

#[test]
fn unaddressed_posting_trailers_are_rejected_before_read_or_merge() {
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        let mut bytes = encoded(codec);
        bytes.insert(bytes.len() - FOOTER_V2_SIZE, 0);
        assert!(BlockPostingList::deserialize(&bytes).is_err());
        let mut output = vec![0xab];
        assert!(BlockPostingList::concatenate_streaming(&[(&bytes, 0)], &mut output).is_err());
        assert_eq!(output, [0xab]);
    }
}

#[test]
fn streaming_merge_rejects_corrupt_blocks_before_any_output() {
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        let valid = encoded(codec);
        let footer = Footer::parse(&valid).unwrap();
        for (at, value) in [(6, 0xe1), (7, 33), (footer.l0_end(), 1)] {
            let mut corrupt = valid.clone();
            corrupt[at] = value;
            assert!(BlockPostingList::deserialize(&corrupt).is_err());
            for sources in [
                vec![(&corrupt[..], 0)],
                vec![(&valid[..], 0), (&corrupt[..], 1000)],
            ] {
                let mut output = vec![0xab];
                let result = BlockPostingList::concatenate_streaming(&sources, &mut output);
                assert!(
                    matches!(result, Err(crate::Error::Corruption(_))),
                    "codec={codec}, byte={at}: {result:?}"
                );
                assert_eq!(
                    output,
                    [0xab],
                    "invalid source must be rejected before writes"
                );
            }
        }
    }
}

#[test]
fn concatenation_rejects_overlapping_or_terminal_document_remapping() {
    let bytes = encoded(PostingCodec::Rounded);
    let list = BlockPostingList::deserialize(&bytes).unwrap();
    for offsets in [
        vec![u32::MAX - 511],
        vec![u32::MAX - 512],
        vec![0, 512],
        vec![1000, 0],
    ] {
        let sources: Vec<_> = offsets.iter().map(|&offset| (&bytes[..], offset)).collect();
        let mut output = vec![0xab];
        let result = BlockPostingList::concatenate_streaming(&sources, &mut output);
        assert!(
            matches!(result, Err(crate::Error::Corruption(_))),
            "offsets={offsets:?}: {result:?}"
        );
        assert_eq!(output, [0xab]);
        let sources: Vec<_> = offsets
            .iter()
            .map(|&offset| (list.clone(), offset))
            .collect();
        assert!(
            BlockPostingList::concatenate_blocks(&sources).is_err(),
            "offsets={offsets:?}"
        );
    }
}

#[test]
fn checked_streaming_merge_matches_encoded_copy_and_materialized_output() {
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        let bytes = encoded(codec);
        let list = BlockPostingList::deserialize(&bytes).unwrap();
        let mut copied = Vec::new();
        BlockPostingList::concatenate_streaming(&[(&bytes, 0)], &mut copied).unwrap();
        assert_eq!(copied, bytes);
        let sources = [(list.clone(), 7), (list.clone(), 1000)];
        let merged = BlockPostingList::concatenate_blocks(&sources).unwrap();
        let mut expected = Vec::new();
        merged.serialize(&mut expected).unwrap();
        let mut output = Vec::new();
        let (count, length) =
            BlockPostingList::concatenate_streaming(&[(&bytes, 7), (&bytes, 1000)], &mut output)
                .unwrap();
        assert_eq!(count, 514);
        assert_eq!(length, output.len());
        assert_eq!(output, expected);
        let reopened = BlockPostingList::deserialize(&output).unwrap();
        assert_eq!(reopened.doc_count(), 514);
    }
}

#[test]
fn concatenation_checks_position_cursor_sum_before_output() {
    let mut bytes = encoded(PostingCodec::Rounded);
    let at = bytes.len() - FOOTER_V2_SIZE + 24;
    bytes[at..at + 8].copy_from_slice(&(1u64 << 63).to_le_bytes());
    let list = BlockPostingList::deserialize(&bytes).unwrap();
    let mut output = vec![0xab];
    let result =
        BlockPostingList::concatenate_streaming(&[(&bytes, 0), (&bytes, 1000)], &mut output);
    assert!(matches!(result, Err(crate::Error::Corruption(_))));
    assert_eq!(output, [0xab]);
    assert!(BlockPostingList::concatenate_blocks(&[(list.clone(), 0), (list, 1000)]).is_err());
}
