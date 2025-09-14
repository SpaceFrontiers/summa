//! Content corruption of posting bytes that passes structural admission:
//! every byte of every codec's serialized list is flipped and every prefix is
//! tried. Each outcome is either a loud deserialize error or a cursor that
//! never panics, never yields an id outside its block's L0 range, and reports
//! any truncation through `decode_block_doc_ids_only` returning `None`.

use hermes_core::structures::{BlockPostingList, PostingCodec, PostingList, TERMINATED};

const CODECS: [PostingCodec; 4] = [
    PostingCodec::Rounded,
    PostingCodec::Packed,
    PostingCodec::Pfor,
    PostingCodec::Simd4x,
];

fn fixture(codec: PostingCodec) -> (Vec<(u32, u32)>, Vec<u8>) {
    let mut docs = Vec::new();
    let mut doc = 3u32;
    for i in 0..300u32 {
        // Mixed gap widths and a few outliers so exact/patched widths and
        // exception tables all appear; tf outliers widen frequency arrays.
        doc += if i % 89 == 0 { 70_000 } else { 1 + i % 9 };
        let tf = if i % 61 == 0 { 40_000 } else { 1 + i % 4 };
        docs.push((doc, tf));
    }
    let mut list = PostingList::new();
    for &(d, tf) in &docs {
        list.push(d, tf);
    }
    let bpl = BlockPostingList::from_posting_list_with_ratio_bounds(
        &list,
        true,
        Some(&|id| id % 50 + 1),
        codec,
    )
    .unwrap();
    let mut bytes = Vec::new();
    bpl.serialize(&mut bytes).unwrap();
    (docs, bytes)
}

/// Traverse an admitted list every way the public cursor allows; return the
/// number of postings the sequential cursor yielded.
fn exercise(list: &BlockPostingList, context: &str) -> usize {
    let blocks = list.num_blocks();
    let bounds: Vec<(u32, u32)> = (0..blocks)
        .map(|b| {
            (
                list.block_first_doc(b).unwrap(),
                list.block_last_doc(b).unwrap(),
            )
        })
        .collect();
    let mut decoded = 0usize;
    let mut first_bad_block = None;
    let mut ids = Vec::new();
    let mut tfs = Vec::new();
    for (block, &(first, last)) in bounds.iter().enumerate() {
        if list.decode_block_into(block, &mut ids, &mut tfs) {
            assert!(
                ids.windows(2).all(|w| w[0] < w[1]),
                "{context}: block {block}"
            );
            assert_eq!(ids.first(), Some(&first), "{context}: block {block}");
            assert_eq!(ids.last(), Some(&last), "{context}: block {block}");
            assert_eq!(ids.len(), tfs.len());
            if first_bad_block.is_none() {
                decoded += ids.len();
            }
        } else if first_bad_block.is_none() {
            first_bad_block = Some(block);
        }
    }
    let mut cursor = list.iterator();
    let mut yielded = 0usize;
    let mut previous = None;
    while cursor.doc() != TERMINATED {
        let doc = cursor.doc();
        let block = cursor.current_block_idx();
        assert!(block < blocks, "{context}");
        assert!(
            (bounds[block].0..=bounds[block].1).contains(&doc),
            "{context}: doc {doc} outside block {block} range {:?}",
            bounds[block]
        );
        assert!(previous.is_none_or(|p| p < doc), "{context}: non-monotone");
        let _ = cursor.term_freq();
        let _ = cursor.position_cursor();
        let _ = cursor.current_block_max_tf();
        previous = Some(doc);
        yielded += 1;
        cursor.advance();
    }
    assert_eq!(cursor.term_freq(), 0);
    match first_bad_block {
        Some(bad) => assert!(
            yielded <= bad * hermes_core::structures::POSTING_BLOCK_SIZE,
            "{context}: cursor must stop at the reported corrupt block {bad}"
        ),
        None => assert_eq!(
            yielded,
            list.doc_count() as usize,
            "{context}: a list with no reported corruption must be complete"
        ),
    }
    assert_eq!(decoded, yielded, "{context}");
    // Seeks: into each block, past each block end, and past the list.
    let mut targets: Vec<u32> = bounds
        .iter()
        .flat_map(|&(first, last)| [first, first.saturating_add(1), last, last.saturating_add(1)])
        .collect();
    targets.push(TERMINATED);
    let mut cursor = list.iterator();
    let mut floor = 0;
    for target in targets {
        let target = target.max(floor);
        let doc = cursor.seek(target);
        assert!(doc >= target, "{context}: seek({target}) returned {doc}");
        if doc != TERMINATED {
            let block = cursor.current_block_idx();
            assert!(
                (bounds[block].0..=bounds[block].1).contains(&doc),
                "{context}: seek {target} landed on {doc} outside block {block}"
            );
            let _ = cursor.term_freq();
        }
        floor = target;
    }
    let mut skipper = list.iterator();
    while skipper.skip_to_next_block() != TERMINATED {}
    yielded
}

#[test]
fn every_single_byte_corruption_is_rejected_or_survives_without_panic_or_silent_truncation() {
    for codec in CODECS {
        let (docs, bytes) = fixture(codec);
        let pristine = BlockPostingList::deserialize(&bytes).unwrap();
        assert_eq!(
            exercise(&pristine, &format!("{codec} pristine")),
            docs.len()
        );
        let mut admitted = 0usize;
        let mut truncated = 0usize;
        for at in 0..bytes.len() {
            for value in [bytes[at] ^ 0xFF, 0, 1, bytes[at] ^ 0x01] {
                if value == bytes[at] {
                    continue;
                }
                let mut corrupt = bytes.clone();
                corrupt[at] = value;
                let context = format!("{codec} byte {at} = {value:#04x}");
                let Ok(list) = BlockPostingList::deserialize(&corrupt) else {
                    continue;
                };
                admitted += 1;
                let yielded = exercise(&list, &context);
                if yielded < docs.len() {
                    truncated += 1;
                }
            }
        }
        // Payload-only flips are admitted structurally, so the content check
        // must be doing real work for every codec.
        assert!(admitted > 0, "{codec}: no flip was admitted");
        assert!(
            truncated > 0,
            "{codec}: some admitted flip must move an id outside its block and be reported"
        );
    }
}

#[test]
fn every_prefix_truncation_is_rejected_or_survives_without_panic() {
    for codec in CODECS {
        let (_, bytes) = fixture(codec);
        for end in 0..bytes.len() {
            let context = format!("{codec} truncated to {end} bytes");
            if let Ok(list) = BlockPostingList::deserialize(&bytes[..end]) {
                exercise(&list, &context);
            }
        }
    }
}
