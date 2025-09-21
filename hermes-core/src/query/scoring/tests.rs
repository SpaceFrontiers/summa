use super::*;

#[test]
fn skewed_conjunction_preserves_frequency_rows_across_seeks_and_partial_batches() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let params = super::super::Bm25Params::default();
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&vec![100; 40001]);
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        let mut common = PostingList::new();
        let mut rare = PostingList::new();
        let mut expected = Vec::new();
        for doc in 0..40001 {
            let tf = doc % 7 + 1;
            if doc % 11 != 1 {
                common.push(doc, tf);
            }
            if doc % 257 == 0 || doc == 40000 {
                rare.push(doc, doc % 5 + 1);
                if doc % 11 != 1 && doc % 13 != 0 {
                    expected.push((
                        doc,
                        params.score(tf as f32, 1.0, 100.0, 100.0)
                            + params.score((doc % 5 + 1) as f32, 2.0, 100.0, 100.0),
                    ));
                }
            }
        }
        let count = expected.len();
        expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let common =
            BlockPostingList::from_posting_list_with_options(&common, false, Some(&|_| 100), codec)
                .unwrap();
        let rare =
            BlockPostingList::from_posting_list_with_options(&rare, false, Some(&|_| 100), codec)
                .unwrap();
        for reversed in [false, true] {
            for k in [1, 10, 128, 200] {
                let mut lists = vec![(common.clone(), 1.0), (rare.clone(), 2.0)];
                if reversed {
                    lists.reverse();
                }
                let make = || {
                    MaxScoreExecutor::text_with_lengths(
                        lists.clone(),
                        100.0,
                        k,
                        Some(LengthSource::Docs(&lengths)),
                        params,
                        1.0,
                    )
                    .require_all_terms()
                    .with_predicate(Box::new(|doc| doc % 13 != 0))
                };
                let (counted, seen) = make().execute_counted_conjunction().unwrap();
                assert_eq!(seen, count as u64);
                for hits in [
                    counted,
                    make().execute_sync().unwrap(),
                    futures::executor::block_on(make().execute()).unwrap(),
                ] {
                    let actual: Vec<_> =
                        hits.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
                    let wanted: Vec<_> = expected
                        .iter()
                        .take(k)
                        .map(|&(doc, score)| (doc, score.to_bits()))
                        .collect();
                    assert_eq!(actual, wanted, "{codec:?} reversed={reversed} k={k}");
                }
            }
        }
    }
}

#[test]
fn ranked_mapped_conjunction_prunes_losers_but_keeps_ties_and_exact_counts() {
    use crate::directories::OwnedBytes;
    use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let num_docs = 8_192u32;
    let params = super::super::Bm25Params::default();
    let mut builder = ChunkMapBuilder::default();
    builder.set_document_units(true);
    for doc in 0..num_docs {
        builder.push(num_docs - 1 - doc, 0, 100).unwrap();
    }
    let mut bytes = Vec::new();
    write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
    let maps = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
    let map = &maps.chunk_maps[&0];
    for high_tf in [1, 32] {
        let mut postings = PostingList::new();
        for doc in 0..num_docs {
            postings.push(doc, if doc < 256 { high_tf } else { 1 });
        }
        let list = BlockPostingList::from_posting_list_with_impact_bounds(
            &postings,
            false,
            Some(&|_| 100),
            PostingCodec::Rounded,
        )
        .unwrap();
        let score = params.score(high_tf as f32, 1.0, 100.0, 100.0)
            + params.score(high_tf as f32, 2.0, 100.0, 100.0);
        for k in [1, 10, 129] {
            let make = || {
                let mut executor = MaxScoreExecutor::text_with_lengths(
                    vec![(list.clone(), 1.0), (list.clone(), 2.0)],
                    100.0,
                    k,
                    Some(LengthSource::Chunks(map)),
                    params,
                    1.0,
                )
                .with_document_map(map)
                .require_all_terms();
                executor.seed_threshold(score * 0.99);
                executor
            };
            assert!(make().can_prune_text_pair());
            let (pruned, visited) = make()
                .run_conjunction::<true>(&mut WindowScratch::default())
                .unwrap();
            let (counted, count) = make().execute_counted_conjunction().unwrap();
            assert_eq!(count, u64::from(num_docs));
            let winners = if high_tf == 1 { num_docs } else { 256 };
            assert_eq!(visited, u64::from(winners));
            let expected: Vec<_> = (num_docs - winners..)
                .take(k)
                .map(|doc| (doc, score.to_bits()))
                .collect();
            for hits in [
                pruned,
                counted,
                make().execute_sync().unwrap(),
                futures::executor::block_on(make().execute()).unwrap(),
            ] {
                let got: Vec<_> = hits
                    .into_iter()
                    .map(|hit| (hit.doc_id, hit.score.to_bits()))
                    .collect();
                assert_eq!(got, expected, "tf={high_tf} k={k}");
            }
            let expired =
                SharedThreshold::for_limit(k).with_deadline(Some(std::time::Instant::now()));
            let (hits, visited) = make()
                .with_budget(Some(expired.clone()))
                .run_conjunction::<true>(&mut WindowScratch::default())
                .unwrap();
            assert!(hits.is_empty());
            assert_eq!(visited, 0);
            assert!(expired.truncated());
        }
    }
}

#[test]
fn mapped_mixed_density_windows_keep_exact_scores_seeded_ties_and_filters() {
    use crate::directories::OwnedBytes;
    use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
    let params = super::super::Bm25Params::default();
    for seed in [1, 9, 17] {
        for terms in [3, 6, 12] {
            let corpus = random_corpus(seed, 6_000, terms);
            let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
            let lists = build_lists(&corpus, Some(&lengths));
            let mut builder = ChunkMapBuilder::default();
            builder.set_document_units(true);
            for doc in 0..corpus.n_docs {
                builder
                    .push(corpus.n_docs - 1 - doc, 0, lengths.length(doc))
                    .unwrap();
            }
            let mut bytes = Vec::new();
            write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
            let maps = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
            let map = &maps.chunk_maps[&0];
            let truth = exhaustive(&corpus, &lists, true, lengths.avg_len(), params);
            for filtered in [false, true] {
                let eligible = move |doc: u32| !filtered || !doc.is_multiple_of(7);
                let mut expected: Vec<_> = truth
                    .iter()
                    .filter(|(doc, _)| eligible(**doc))
                    .map(|(&doc, &score)| (map.doc_id(doc), score))
                    .collect();
                expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                for k in [1, 10, 1_000] {
                    let wanted: Vec<_> = expected
                        .iter()
                        .take(k)
                        .map(|&(doc, score)| (doc, score.to_bits()))
                        .collect();
                    for seeded in [false, true] {
                        let make = || {
                            let mut executor = MaxScoreExecutor::text_with_lengths(
                                lists.clone(),
                                lengths.avg_len(),
                                k,
                                Some(LengthSource::Chunks(map)),
                                params,
                                1.0,
                            )
                            .with_document_map(map)
                            .with_predicate(Box::new(eligible));
                            if seeded {
                                executor.seed_threshold(expected[k - 1].1);
                            }
                            executor
                        };
                        for hits in [
                            make().execute_sync().unwrap(),
                            futures::executor::block_on(make().execute()).unwrap(),
                        ] {
                            let got: Vec<_> = hits
                                .into_iter()
                                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                                .collect();
                            assert_eq!(
                                got, wanted,
                                "seed={seed} terms={terms} k={k} filtered={filtered} seeded={seeded}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mapped_windows_preserve_exhaustive_bits_for_gaps_ties_filters_and_late_winners() {
    use crate::directories::OwnedBytes;
    use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    use std::collections::BTreeMap;
    let num_docs = 8192;
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(
        &(0..num_docs)
            .map(|doc| [0, 1, 97, 65535][doc % 4])
            .collect::<Vec<u16>>(),
    );
    for num_terms in [2u32, 3] {
        for layout in 0..4 {
            let source: Vec<Vec<(u32, u32)>> = (0..num_terms)
                .map(|term| {
                    (0..num_docs as u32)
                        .filter(|&doc| match layout {
                            0 => doc % (term + 2) == 0,
                            1 => {
                                if term == 0 {
                                    doc < 4000 && doc % 2 == 0
                                } else {
                                    doc >= 4096 && doc % 3 == 0
                                }
                            }
                            2 => doc % 7 == 0,
                            _ => term == 1 && doc % 5 == 0,
                        })
                        .map(|doc| (doc, if doc > 8100 { 1000 } else { doc % 5 + 1 }))
                        .collect()
                })
                .collect();
            for codec in [
                PostingCodec::Rounded,
                PostingCodec::Packed,
                PostingCodec::Pfor,
                PostingCodec::Simd4x,
            ] {
                for with_lengths in [false, true] {
                    let mut builder = ChunkMapBuilder::default();
                    builder.set_document_units(true);
                    for doc in 0..num_docs as u32 {
                        builder
                            .push(
                                num_docs as u32 - 1 - doc,
                                0,
                                if with_lengths { lengths.length(doc) } else { 0 },
                            )
                            .unwrap();
                    }
                    let mut bytes = Vec::new();
                    write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
                    let maps = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
                    let map = &maps.chunk_maps[&0];
                    for params in [
                        super::super::Bm25Params::default(),
                        super::super::Bm25Params { k1: 0.0, b: 1.0 },
                    ] {
                        for reverse in [false, true] {
                            let mut order: Vec<_> = (0..num_terms as usize).collect();
                            if reverse {
                                order.reverse();
                            }
                            let lists: Vec<_> = order
                                .iter()
                                .map(|&term| {
                                    let mut postings = PostingList::new();
                                    for &(doc, tf) in &source[term] {
                                        postings.push(doc, tf);
                                    }
                                    let list = if with_lengths {
                                        BlockPostingList::from_posting_list_with_impact_bounds(
                                            &postings,
                                            false,
                                            Some(&|doc| lengths.length(doc).max(1)),
                                            codec,
                                        )
                                        .unwrap()
                                    } else {
                                        BlockPostingList::from_posting_list_with_codec(
                                            &postings, codec,
                                        )
                                        .unwrap()
                                    };
                                    (list, [1.2345, 9.876, 0.03125][term])
                                })
                                .collect();
                            let mut expected = BTreeMap::<u32, f32>::new();
                            for &term in &order {
                                for &(doc, frequency) in &source[term] {
                                    let tf = frequency as f32;
                                    let len = if with_lengths { lengths.length(doc) } else { 0 };
                                    *expected.entry(doc).or_default() += params.score(
                                        tf,
                                        [1.2345, 9.876, 0.03125][term],
                                        if len == 0 { tf } else { len as f32 },
                                        17.5,
                                    );
                                }
                            }
                            for mapped in [false, true] {
                                for k in [0, 1, 10, 1000, 9000] {
                                    for filtered in [false, true] {
                                        let eligible = |doc| !filtered || doc % 11 == 0;
                                        let mut wanted: Vec<_> = expected
                                            .iter()
                                            .filter(|(doc, _)| eligible(**doc))
                                            .map(|(&doc, &score)| {
                                                (if mapped { map.doc_id(doc) } else { doc }, score)
                                            })
                                            .collect();
                                        wanted.sort_unstable_by(|a, b| {
                                            b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
                                        });
                                        wanted.truncate(k);
                                        let make = || {
                                            let executor = MaxScoreExecutor::text_with_lengths(
                                                lists.clone(),
                                                17.5,
                                                k,
                                                if mapped {
                                                    Some(LengthSource::Chunks(map))
                                                } else {
                                                    with_lengths
                                                        .then_some(LengthSource::Docs(&lengths))
                                                },
                                                params,
                                                1.0,
                                            )
                                            .with_predicate(Box::new(eligible));
                                            if mapped {
                                                executor.with_document_map(map)
                                            } else {
                                                executor
                                            }
                                        };
                                        let bits = |hits: Vec<ScoredDoc>| {
                                            hits.into_iter()
                                                .map(|h| (h.doc_id, h.score.to_bits()))
                                                .collect::<Vec<_>>()
                                        };
                                        let wanted: Vec<_> = wanted
                                            .into_iter()
                                            .map(|(doc, score)| (doc, score.to_bits()))
                                            .collect();
                                        assert_eq!(
                                            bits(make().execute_sync().unwrap()),
                                            wanted,
                                            "mapped={mapped} layout={layout} {codec:?} norms={with_lengths} k={k} filtered={filtered} reverse={reverse} params={params:?}"
                                        );
                                        assert_eq!(
                                            bits(
                                                futures::executor::block_on(make().execute())
                                                    .unwrap()
                                            ),
                                            wanted
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn two_term_windows_keep_seeded_ties_and_observe_predicate_deadlines() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let mut postings = PostingList::new();
    for doc in 0..389 {
        postings.push(u32::MAX - 10_000 + doc * 2, 1);
    }
    let list =
        BlockPostingList::from_posting_list_with_codec(&postings, PostingCodec::Rounded).unwrap();
    let params = super::super::Bm25Params { k1: 0.0, b: 1.0 };
    let make = || {
        MaxScoreExecutor::text(
            vec![(list.clone(), 2.0), (list.clone(), 3.0)],
            1.0,
            10,
            None,
            params,
            1.0,
        )
    };
    let expected: Vec<_> = (0..10)
        .map(|i| (u32::MAX - 10_000 + i * 2, 5.0f32.to_bits()))
        .collect();
    for floor in [0.0, 4.99, 5.0, 5.1] {
        let mut executor = make();
        executor.seed_threshold(floor);
        let actual: Vec<_> = executor
            .execute_sync()
            .unwrap()
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect();
        assert_eq!(
            actual,
            if floor <= 5.0 {
                expected.clone()
            } else {
                Vec::new()
            }
        );
    }
    let budget = SharedThreshold::for_limit(10).with_deadline(Some(std::time::Instant::now()));
    assert!(
        make()
            .with_budget(Some(budget.clone()))
            .with_predicate(Box::new(|_| panic!("expired predicate")))
            .execute_sync()
            .unwrap()
            .is_empty()
    );
    assert!(budget.truncated());
}

#[test]
fn text_cursor_seeks_preserve_postings_and_scores_across_lazy_block_boundaries() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        for len in [0, 1, 127, 128, 129, 389, 2049] {
            for base in [0, u32::MAX - 50_000] {
                let rows: Vec<_> = (0..len)
                    .map(|i| {
                        (
                            base + i * 7 + i / 32 * 17,
                            [1, 3, 127, 65_536, u32::MAX][i as usize % 5],
                        )
                    })
                    .collect();
                let mut postings = PostingList::new();
                for &(doc, tf) in &rows {
                    postings.push(doc, tf);
                }
                let list =
                    BlockPostingList::from_posting_list_with_codec(&postings, codec).unwrap();
                let params = super::super::Bm25Params::default();
                for asynchronous in [false, true] {
                    for stride in [1, 13, 127, 128, 509] {
                        let mut cursor =
                            TermCursor::text_with_params(list.clone(), 1.2345, 17.5, None, params);
                        let mut expected = 0;
                        for i in (0..rows.len()).step_by(stride) {
                            let target = rows[i].0.saturating_sub((i % 3) as u32);
                            expected += rows[expected..].partition_point(|row| row.0 < target);
                            let wanted = rows.get(expected).map_or(u32::MAX, |row| row.0);
                            let found = if asynchronous {
                                futures::executor::block_on(cursor.seek(target)).unwrap()
                            } else {
                                cursor.seek_sync(target).unwrap()
                            };
                            assert_eq!(
                                found, wanted,
                                "{codec:?} len={len} base={base} stride={stride} i={i}"
                            );
                            assert_eq!(cursor.doc(), wanted);
                            // Equal/backward seeks must preserve the current posting.
                            for earlier in [target, target.saturating_sub(100)] {
                                let found = if asynchronous {
                                    futures::executor::block_on(cursor.seek(earlier)).unwrap()
                                } else {
                                    cursor.seek_sync(earlier).unwrap()
                                };
                                assert_eq!(found, wanted);
                            }
                            if i % 7 == 0 && wanted != u32::MAX {
                                if asynchronous {
                                    futures::executor::block_on(cursor.ensure_block_loaded())
                                        .unwrap();
                                } else {
                                    cursor.ensure_block_loaded_sync().unwrap();
                                }
                                cursor.ensure_scores();
                                let tf = rows[expected].1 as f32;
                                assert_eq!(
                                    cursor.score().to_bits(),
                                    params.score(tf, 1.2345, tf, 17.5).to_bits()
                                );
                            }
                            if i % 5 == 0 {
                                expected = (expected + 1).min(rows.len());
                                let wanted = rows.get(expected).map_or(u32::MAX, |row| row.0);
                                let found = if asynchronous {
                                    futures::executor::block_on(cursor.advance()).unwrap()
                                } else {
                                    cursor.advance_sync().unwrap()
                                };
                                assert_eq!(found, wanted);
                                // Keep the next-block payload deferred until the next seek.
                                assert_eq!(cursor.doc(), wanted);
                            }
                        }
                        for target in [u32::MAX, 0, u32::MAX] {
                            let found = if asynchronous {
                                futures::executor::block_on(cursor.seek(target)).unwrap()
                            } else {
                                cursor.seek_sync(target).unwrap()
                            };
                            assert_eq!(found, u32::MAX);
                            assert_eq!(cursor.doc(), u32::MAX);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn competitive_group_skips_keep_future_terms_and_do_not_decode_rejected_payloads() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let make = |base: u32| {
        let mut postings = PostingList::new();
        for doc in 0..2048 {
            postings.push(base + doc, 1);
        }
        BlockPostingList::from_posting_list_with_impact_bounds(
            &postings,
            false,
            Some(&|_| 17),
            PostingCodec::Rounded,
        )
        .unwrap()
    };
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&[17; 2048]);
    let params = super::super::Bm25Params::default();
    let first = TermCursor::text_with_params(
        make(0),
        1.0,
        17.0,
        Some(LengthSource::Docs(&lengths)),
        params,
    );
    let future = u32::MAX - 4096;
    let second = TermCursor::text_with_params(
        make(future),
        1000.0,
        17.0,
        Some(LengthSource::Docs(&lengths)),
        params,
    );
    let mut executor = MaxScoreExecutor::new(vec![first, second], 1, 1.0);
    let mut bounds = [0.0; 2];
    let mut checked = None;
    assert!(
        executor
            .skip_noncompetitive_group(0, 127, 100.0, &mut bounds, &mut checked)
            .unwrap()
    );
    assert_eq!(checked, Some(1023));
    assert_eq!(executor.cursors[0].doc(), 1024);
    assert_eq!(executor.cursors[1].doc(), future);
    assert!(
        executor
            .cursors
            .iter()
            .all(|c| !c.block_loaded && c.doc_ids.is_empty())
    );
    assert!(
        !executor
            .skip_noncompetitive_group(1024, 1151, 0.0, &mut bounds, &mut checked)
            .unwrap()
    );
    assert_eq!(executor.cursors[0].doc(), 1024);
    executor.cursors[0].skip_past_sync(2047).unwrap();
    assert!(
        !executor
            .skip_noncompetitive_group(future, future + 127, 100.0, &mut bounds, &mut checked)
            .unwrap()
    );
    assert_eq!(executor.cursors[1].doc(), future);
    assert!(
        executor
            .skip_noncompetitive_group(future, future + 127, 10_000.0, &mut bounds, &mut checked)
            .unwrap()
    );
    assert_eq!(executor.cursors[1].doc(), future + 1024);
    assert!(!executor.cursors[1].block_loaded);
}

// ── Windowed executor parity ─────────────────────────────────────────

struct Corpus {
    /// Per term: sorted `(doc, tf)` postings.
    postings: Vec<Vec<(u32, u32)>>,
    lengths: Vec<u16>,
    n_docs: u32,
}

#[test]
fn candidate_runs_preserve_score_bits_when_switching_to_and_from_full_block_scores() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let mut postings = PostingList::new();
    let frequency = |doc: u32| [1, 3, 127, 65535, 65536, u32::MAX][doc as usize % 7 % 6];
    for doc in (0..2000).step_by(3) {
        postings.push(doc, frequency(doc));
    }
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(
        &(0..2000)
            .map(|doc| [0, 1, 97, 65535][doc % 4])
            .collect::<Vec<u16>>(),
    );
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        let list = BlockPostingList::from_posting_list_with_codec(&postings, codec).unwrap();
        for with_lengths in [false, true] {
            for required in [false, true] {
                for params in [
                    super::super::Bm25Params::default(),
                    super::super::Bm25Params { k1: 0.0, b: 1.0 },
                    super::super::Bm25Params { k1: 4.0, b: 0.0 },
                ] {
                    let mut cursor = TermCursor::text_with_params(
                        list.clone(),
                        1.2345,
                        17.5,
                        with_lengths.then_some(LengthSource::Docs(&lengths)),
                        params,
                    );
                    let score = |doc: u32| {
                        let tf = frequency(doc) as f32;
                        let len = if with_lengths {
                            lengths.length(doc) as f32
                        } else {
                            0.0
                        };
                        params.score(tf, 1.2345, if len == 0.0 { tf } else { len }, 17.5)
                    };
                    for (round, input) in [
                        vec![0, 1, 3, 10, 27],
                        vec![30, 34, 300, 381],
                        vec![384, 390, 600, 777, 999],
                        vec![1002, 1024, 1800, 1998, 2001],
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let expected: Vec<_> = input
                            .iter()
                            .copied()
                            .filter_map(|doc| {
                                let matched = doc < 2000 && doc % 3 == 0;
                                (matched || !required).then(|| {
                                    (
                                        doc,
                                        (0.25 + if matched { score(doc) } else { 0.0 }).to_bits(),
                                    )
                                })
                            })
                            .collect();
                        let mut docs = input;
                        let mut scores = vec![0.25; docs.len()];
                        let mut contributions = vec![0.0; 2048];
                        let mut mask = vec![0; 32];
                        assert!(
                            cursor
                                .score_candidates_sync(
                                    0,
                                    &mut docs,
                                    &mut scores,
                                    required,
                                    Some((&mut contributions, &mut mask)),
                                    None
                                )
                                .unwrap()
                        );
                        let actual: Vec<_> = docs
                            .into_iter()
                            .zip(scores.into_iter().map(f32::to_bits))
                            .collect();
                        assert_eq!(
                            actual, expected,
                            "codec={codec:?} lengths={with_lengths} required={required} params={params:?} round={round}"
                        );
                        for &(doc, _) in &expected {
                            let present = doc < 2000 && doc % 3 == 0;
                            assert_eq!(mask[doc as usize / 64] & (1 << (doc % 64)) != 0, present);
                            if present {
                                assert_eq!(
                                    contributions[doc as usize].to_bits(),
                                    score(doc).to_bits()
                                );
                            }
                        }
                        // The first run computes only selected scores. A later
                        // full window must reuse decoded TFs and compute every
                        // remaining posting correctly; the next run reuses it.
                        if !cursor.exhausted {
                            cursor.ensure_scores();
                            for (&doc, &actual) in cursor.doc_ids.iter().zip(&cursor.scores) {
                                assert_eq!(actual.to_bits(), score(doc).to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn score_required_candidate_intersection_defers_scores_and_observes_cancellation() {
    let mut postings = crate::structures::PostingList::new();
    for doc in (0..8192).step_by(2) {
        postings.push(doc, 1);
    }
    let list = crate::structures::BlockPostingList::from_posting_list(&postings).unwrap();
    let mut cursor =
        TermCursor::text_with_params(list, 1.0, 1.0, None, super::super::Bm25Params::default());
    let mut docs = vec![1, 257, 4097];
    let mut scores = vec![0.0; docs.len()];
    assert!(
        cursor
            .score_candidates_sync(0, &mut docs, &mut scores, true, None, None)
            .unwrap()
    );
    assert!(docs.is_empty());
    assert!(scores.is_empty());
    assert!(
        cursor.scores.is_empty(),
        "rejected candidates decoded TFs and scores"
    );
    let before = cursor.doc();
    let budget = SharedThreshold::for_limit(1).with_deadline(Some(std::time::Instant::now()));
    docs.push(7000);
    scores.push(2.0);
    assert!(
        !cursor
            .score_candidates_sync(0, &mut docs, &mut scores, true, None, Some(&budget))
            .unwrap()
    );
    assert!(budget.truncated());
    assert_eq!(cursor.doc(), before);
    assert_eq!(docs, [7000]);
    assert_eq!(scores, [2.0]);
}

#[test]
fn score_required_windows_preserve_bits_across_empty_intersections_and_late_winners() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let n_docs = 8192;
    let terms: Vec<Vec<(u32, u32)>> = (0..3)
        .map(|term| {
            (0..n_docs)
                .filter(|&doc| {
                    !(16..=8100).contains(&doc)
                        || match term {
                            0 => doc % 17 == 0,
                            1 => doc % 3 == 0 && (!(4096..7500).contains(&doc) || doc % 17 != 0),
                            _ => doc % 2 == 0,
                        }
                })
                .map(|doc| (doc, if doc < 16 { 1 } else { 1 + doc % 3 }))
                .collect()
        })
        .collect();
    let params = super::super::Bm25Params {
        b: 0.0,
        ..Default::default()
    };
    for order in [
        vec![0, 1, 2],
        vec![2, 1, 0],
        vec![1, 0, 2],
        vec![0, 1, 0, 2],
    ] {
        let corpus = Corpus {
            postings: order.iter().map(|&i| terms[i].clone()).collect(),
            lengths: vec![100; n_docs as usize],
            n_docs,
        };
        let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        for layout in 0..3 {
            let lists: Vec<_> = corpus
                .postings
                .iter()
                .zip(&order)
                .map(|(postings, &term)| {
                    let mut list = PostingList::new();
                    for &(doc, tf) in postings {
                        list.push(doc, tf);
                    }
                    let list = match layout {
                        0 => BlockPostingList::from_posting_list_with(&list, false, Some(&|_| 100)),
                        1 => BlockPostingList::from_posting_list_with_ratio_bounds(
                            &list,
                            false,
                            Some(&|_| 100),
                            PostingCodec::Rounded,
                        ),
                        _ => BlockPostingList::from_posting_list_with_impact_bounds(
                            &list,
                            false,
                            Some(&|_| 100),
                            PostingCodec::Rounded,
                        ),
                    }
                    .unwrap();
                    (list, if term == 2 { 0.1 } else { 1.0 })
                })
                .collect();
            let truth = exhaustive(&corpus, &lists, true, 100.0, params);
            for k in [1, 10, 100, 1000] {
                for predicate in [false, true] {
                    let mut expected: Vec<_> = truth
                        .iter()
                        .filter(|(doc, _)| !predicate || *doc % 5 != 0)
                        .map(|(&doc, &score)| (doc, score.to_bits()))
                        .collect();
                    expected.sort_by(|a, b| {
                        f32::from_bits(b.1)
                            .total_cmp(&f32::from_bits(a.1))
                            .then(a.0.cmp(&b.0))
                    });
                    expected.truncate(k);
                    let mut executor = MaxScoreExecutor::text(
                        lists.clone(),
                        100.0,
                        k,
                        Some(&lengths),
                        params,
                        1.0,
                    );
                    if predicate {
                        executor = executor.with_predicate(Box::new(|doc| doc % 5 != 0));
                    }
                    let actual: Vec<_> = executor
                        .execute_windowed()
                        .unwrap()
                        .into_iter()
                        .map(|hit| (hit.doc_id, hit.score.to_bits()))
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "layout={layout},order={order:?},k={k},predicate={predicate}"
                    );
                }
            }
        }
    }
}

#[test]
fn sparse_posting_windows_preserve_canonical_bits_with_absent_terms_and_duplicate_clauses() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let n_docs = 400_000;
    let mut terms: Vec<Vec<(u32, u32)>> = (0..3)
        .map(|term| {
            (0..389)
                .map(|i| {
                    let doc = i * 1000 + 13 + if i % 7 == 0 { 0 } else { term * 7 };
                    (doc, if i > 370 { 1 + i % 41 } else { 1 + i % 3 })
                })
                .collect()
        })
        .collect();
    terms.push(Vec::new());
    for order in [
        vec![0, 1, 2],
        vec![2, 0, 2, 1],
        vec![0, 3, 1],
        (0..64).map(|i| i % 3).collect(),
    ] {
        let corpus = Corpus {
            postings: order.iter().map(|&i| terms[i].clone()).collect(),
            lengths: vec![100; n_docs as usize],
            n_docs,
        };
        let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let lists: Vec<_> = corpus
                .postings
                .iter()
                .zip(&order)
                .map(|(postings, &term)| {
                    let mut list = PostingList::new();
                    for &(doc, tf) in postings {
                        list.push(doc, tf);
                    }
                    let list = BlockPostingList::from_posting_list_with_options(
                        &list,
                        false,
                        Some(&|_| 100),
                        codec,
                    )
                    .unwrap();
                    (list, if term == 2 { 0.1 } else { 1.0 })
                })
                .collect();
            for params in [
                super::super::Bm25Params::default(),
                super::super::Bm25Params { k1: 0.0, b: 1.0 },
            ] {
                for real_lengths in [false, true] {
                    let truth = exhaustive(&corpus, &lists, real_lengths, 100.0, params);
                    for k in [0, 1, 10, 1000] {
                        for predicate in [false, true] {
                            let mut expected: Vec<_> = truth
                                .iter()
                                .filter(|(doc, _)| !predicate || *doc % 5 != 0)
                                .map(|(&doc, &score)| (doc, score.to_bits()))
                                .collect();
                            expected.sort_by(|a, b| {
                                f32::from_bits(b.1)
                                    .total_cmp(&f32::from_bits(a.1))
                                    .then(a.0.cmp(&b.0))
                            });
                            expected.truncate(k);
                            let make = || {
                                let mut e = MaxScoreExecutor::text(
                                    lists.clone(),
                                    100.0,
                                    k,
                                    real_lengths.then_some(&lengths),
                                    params,
                                    1.0,
                                );
                                if predicate {
                                    e = e.with_predicate(Box::new(|doc| doc % 5 != 0));
                                }
                                e
                            };
                            let actual: Vec<_> = make()
                                .execute_sync()
                                .unwrap()
                                .into_iter()
                                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                                .collect();
                            assert_eq!(
                                actual, expected,
                                "codec={codec:?}, order={order:?}, k={k}, predicate={predicate}, lengths={real_lengths}"
                            );
                            if order.len() <= 4 {
                                let sync: Vec<_> = make()
                                    .execute_sync()
                                    .unwrap()
                                    .into_iter()
                                    .map(|hit| (hit.doc_id, hit.score.to_bits()))
                                    .collect();
                                let asynchronous: Vec<_> =
                                    futures::executor::block_on(make().execute())
                                        .unwrap()
                                        .into_iter()
                                        .map(|hit| (hit.doc_id, hit.score.to_bits()))
                                        .collect();
                                assert_eq!(sync, expected);
                                assert_eq!(asynchronous, expected);
                            }
                        }
                    }
                }
            }
            let budget =
                SharedThreshold::for_limit(10).with_deadline(Some(std::time::Instant::now()));
            let expired =
                MaxScoreExecutor::text(lists, 100.0, 10, Some(&lengths), Default::default(), 1.0)
                    .with_budget(Some(budget.clone()))
                    .with_predicate(Box::new(|_| panic!("expired predicate was called")));
            assert!(expired.execute_sync().unwrap().is_empty());
            assert!(budget.truncated());
        }
    }
}

#[test]
fn retained_union_hits_survive_conjunction_tail_exhaustion_and_deadline() {
    let corpus = random_corpus(9, 300, 2);
    let lists = build_lists(&corpus, None);
    for expired in [false, true] {
        let budget =
            SharedThreshold::for_limit(10).with_deadline(expired.then(std::time::Instant::now));
        let mut executor =
            MaxScoreExecutor::text(lists.clone(), 1.0, 10, None, Default::default(), 1.0)
                .with_budget(Some(budget.clone()));
        executor.collector.insert(7, 42.0);
        if !expired {
            executor.cursors[0].exhausted = true;
        }
        let (hits, count) = executor
            .run_conjunction::<false>(&mut WindowScratch::default())
            .unwrap();
        assert_eq!(count, 0);
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            vec![(7, 42.0f32.to_bits())]
        );
        assert_eq!(budget.truncated(), expired);
    }
}

#[cfg(feature = "query-diagnostics")]
#[test]
fn mapped_union_intersects_after_single_terms_cannot_reach_the_heap() {
    use crate::directories::OwnedBytes;
    use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    for arity in [2u32, 3, 5] {
        for overlap in [false, true] {
            for n in [128u32, 256, 4096] {
                let mut builder = ChunkMapBuilder::default();
                builder.set_document_units(true);
                for slot in 0..n {
                    builder.push(n - 1 - slot, 0, 400).unwrap();
                }
                let mut bytes = Vec::new();
                write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
                let maps = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
                let map = &maps.chunk_maps[&0];
                let lists: Vec<_> = (0..arity)
                    .map(|term| {
                        let mut list = PostingList::new();
                        for slot in 0..n {
                            if (overlap && (slot < 256 || slot >= n - 128)) || slot % arity == term
                            {
                                list.push(slot, if slot >= n - 128 { 200 } else { 100 });
                            }
                        }
                        (
                            BlockPostingList::from_posting_list_with_ratio_bounds(
                                &list,
                                false,
                                Some(&|_| 400),
                                PostingCodec::Rounded,
                            )
                            .unwrap(),
                            1.0,
                        )
                    })
                    .collect();
                let make = || {
                    MaxScoreExecutor::text_with_lengths(
                        lists.clone(),
                        400.0,
                        10,
                        Some(LengthSource::Chunks(map)),
                        Default::default(),
                        1.0,
                    )
                    .with_document_map(map)
                };
                let (actual, work) =
                    crate::search_diagnostics::capture_sync(|| make().execute_sync());
                let actual = actual.unwrap();
                assert_eq!(
                    actual.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
                    (0..10).collect::<Vec<_>>()
                );
                let mut expected = MaxScoreExecutor::text_with_lengths(
                    lists.clone(),
                    400.0,
                    n as usize,
                    Some(LengthSource::Chunks(map)),
                    Default::default(),
                    1.0,
                )
                .execute_doc_at_a_time_sync()
                .unwrap();
                for hit in &mut expected {
                    hit.doc_id = map.doc_id(hit.doc_id);
                }
                expected.sort_unstable_by(|a, b| {
                    b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id))
                });
                expected.truncate(10);
                let values = |hits: &[ScoredDoc]| {
                    hits.iter()
                        .map(|r| (r.doc_id, r.score.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(values(&actual), values(&expected));
                assert!(
                    !overlap || work.exact_score_units <= 512 * u64::from(arity),
                    "single-term non-winners must not be scored after the heap proves both terms necessary: {work:?}"
                );
            }
        }
    }
}

#[test]
fn semantic_conjunction_batches_preserve_bits_across_empty_intersections_and_late_winners() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let n_docs = 8192;
    let terms: Vec<Vec<(u32, u32)>> = (0..3)
        .map(|term| {
            (0..n_docs)
                .filter(|&doc| {
                    !(16..=8100).contains(&doc)
                        || match term {
                            0 => doc % 17 == 0,
                            1 => doc % 3 == 0 && (!(4096..7500).contains(&doc) || doc % 17 != 0),
                            _ => doc % 2 == 0,
                        }
                })
                .map(|doc| (doc, if doc < 16 { 1 } else { 1 + doc % 3 }))
                .collect()
        })
        .collect();
    let params = super::super::Bm25Params {
        b: 0.0,
        ..Default::default()
    };
    for order in [
        vec![0, 1, 2],
        vec![2, 1, 0],
        vec![1, 0, 2],
        vec![0, 1, 0, 2],
        (0..64).map(|i| i % 3).collect(),
    ] {
        let corpus = Corpus {
            postings: order.iter().map(|&i| terms[i].clone()).collect(),
            lengths: vec![100; n_docs as usize],
            n_docs,
        };
        let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        for layout in 0..3 {
            let lists: Vec<_> = corpus
                .postings
                .iter()
                .zip(&order)
                .map(|(postings, &term)| {
                    let mut list = PostingList::new();
                    for &(doc, tf) in postings {
                        list.push(doc, tf);
                    }
                    let list = match layout {
                        0 => BlockPostingList::from_posting_list_with(&list, false, Some(&|_| 100)),
                        1 => BlockPostingList::from_posting_list_with_ratio_bounds(
                            &list,
                            false,
                            Some(&|_| 100),
                            PostingCodec::Rounded,
                        ),
                        _ => BlockPostingList::from_posting_list_with_impact_bounds(
                            &list,
                            false,
                            Some(&|_| 100),
                            PostingCodec::Rounded,
                        ),
                    }
                    .unwrap();
                    (list, if term == 2 { 0.1 } else { 1.0 })
                })
                .collect();
            let truth = exhaustive(&corpus, &lists, true, 100.0, params);
            for k in [1, 10, 100, 1000] {
                for predicate in [false, true] {
                    let mut expected: Vec<_> = truth
                        .iter()
                        .filter(|(doc, _)| {
                            (!predicate || *doc % 5 != 0)
                                && corpus.postings.iter().all(|postings| {
                                    postings.binary_search_by_key(*doc, |&(id, _)| id).is_ok()
                                })
                        })
                        .map(|(&doc, &score)| (doc, score.to_bits()))
                        .collect();
                    expected.sort_by(|a, b| {
                        f32::from_bits(b.1)
                            .total_cmp(&f32::from_bits(a.1))
                            .then(a.0.cmp(&b.0))
                    });
                    expected.truncate(k);
                    let mut executor = MaxScoreExecutor::text(
                        lists.clone(),
                        100.0,
                        k,
                        Some(&lengths),
                        params,
                        1.0,
                    )
                    .require_all_terms();
                    if predicate {
                        executor = executor.with_predicate(Box::new(|doc| doc % 5 != 0));
                    }
                    let actual: Vec<_> = executor
                        .execute_sync()
                        .unwrap()
                        .into_iter()
                        .map(|hit| (hit.doc_id, hit.score.to_bits()))
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "layout={layout},order={order:?},k={k},predicate={predicate}"
                    );
                }
            }
        }
    }
}

#[test]
fn windowed_text_preserves_exhaustive_score_order_and_document_ties() {
    let corpus = random_corpus(71, 8192, 4);
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
    let lists = build_lists(&corpus, Some(&lengths));
    let avg = corpus.lengths.iter().map(|&n| n as f32).sum::<f32>() / corpus.n_docs as f32;
    let params = super::super::Bm25Params::default();
    let scores = exhaustive(&corpus, &lists, true, avg, params);
    let mut expected: Vec<_> = scores.into_iter().collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    for k in [10, 100, 1000] {
        let actual = MaxScoreExecutor::text(lists.clone(), avg, k, Some(&lengths), params, 1.0)
            .execute_sync()
            .unwrap();
        assert_eq!(actual.len(), expected.len().min(k));
        for (rank, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                (actual.doc_id, actual.score),
                *expected,
                "rank {rank}, k={k}"
            );
        }
    }
}

#[test]
fn sparse_text_windows_do_not_reuse_a_previous_documents_scores() {
    for term_count in [1, 4] {
        let n_docs = 32_768;
        let corpus = Corpus {
            postings: (0..term_count)
                .map(|term| {
                    (0..n_docs)
                        .filter(|&doc| doc % 1024 == (doc / 4096 + term) % 4)
                        .map(|doc| (doc, doc % 11 + 1))
                        .collect()
                })
                .collect(),
            lengths: vec![100; n_docs as usize],
            n_docs,
        };
        let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        let lists = build_lists(&corpus, Some(&lengths));
        let params = super::super::Bm25Params::default();
        let mut expected: Vec<_> = exhaustive(&corpus, &lists, true, 100.0, params)
            .into_iter()
            .collect();
        expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for k in [10, 100, 1000] {
            let actual: Vec<_> =
                MaxScoreExecutor::text(lists.clone(), 100.0, k, Some(&lengths), params, 1.0)
                    .execute_sync()
                    .unwrap()
                    .into_iter()
                    .map(|hit| (hit.doc_id, hit.score))
                    .collect();
            let bits = |hits: &[(u32, f32)]| {
                hits.iter()
                    .map(|&(doc, score)| (doc, score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&actual),
                bits(&expected[..expected.len().min(k)]),
                "k={k}, terms={term_count}"
            );
        }
    }
}

#[test]
fn sparse_candidate_runs_preserve_gaps_frequencies_and_terminal_ids() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let source: Vec<_> = (0..391u32)
        .map(|i| (i * 53, i % 9))
        .chain([(u32::MAX - 4095, 1), (u32::MAX - 1, 37)])
        .collect();
    let params = super::super::Bm25Params::default();
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        let mut postings = PostingList::new();
        for &(doc, tf) in &source {
            postings.push(doc, tf);
        }
        let list = BlockPostingList::from_posting_list_with_codec(&postings, codec).unwrap();
        let mut cursor = TermCursor::text_with_params(list, 1.2345, 17.5, None, params);
        let mut values = vec![f32::NAN; WINDOW_IDS];
        let mut present = vec![0; WINDOW_IDS / 64];
        let mut docs = Vec::with_capacity(WINDOW_IDS);
        let mut scores = Vec::with_capacity(WINDOW_IDS);
        for (from, to) in (0..6u32)
            .map(|i| (i * 4096, i * 4096 + 4095))
            .chain([(u32::MAX - 4095, u32::MAX - 1)])
        {
            docs.clear();
            scores.clear();
            present.fill(0);
            cursor.seek_sync(from).unwrap();
            cursor
                .append_scored_window_sync(
                    from,
                    to,
                    &mut docs,
                    &mut scores,
                    Some((&mut values, &mut present)),
                )
                .unwrap();
            let expected: Vec<_> = source
                .iter()
                .copied()
                .filter(|&(doc, _)| (from..=to).contains(&doc))
                .collect();
            assert_eq!(
                docs,
                expected.iter().map(|&(doc, _)| doc).collect::<Vec<_>>(),
                "{codec:?} {from}"
            );
            for ((&doc, &score), &(_, tf)) in docs.iter().zip(&scores).zip(&expected) {
                let expected = 0.0 + params.score(tf as f32, 1.2345, tf as f32, 17.5);
                assert_eq!(score.to_bits(), expected.to_bits(), "{codec:?} {doc}");
                let slot = (doc - from) as usize;
                assert_ne!(present[slot >> 6] & (1u64 << (slot & 63)), 0);
                assert_eq!(values[slot].to_bits(), expected.to_bits());
            }
            assert_eq!(
                present
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>(),
                expected.len()
            );
            assert_eq!(
                cursor.doc(),
                source
                    .iter()
                    .find(|&&(doc, _)| doc > to)
                    .map_or(u32::MAX, |&(doc, _)| doc)
            );
        }
    }
}

#[test]
fn required_windows_preserve_optional_membership_zero_tf_ties_and_late_winners() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let base = random_corpus(873, 8193, 4);
    for order in [vec![0, 1], vec![3, 0, 1, 2], vec![0, 3, 2, 0, 1]] {
        let corpus = Corpus {
            n_docs: base.n_docs,
            lengths: base.lengths.clone(),
            postings: order
                .iter()
                .map(|&i| {
                    base.postings[i]
                        .iter()
                        .map(|&(doc, tf)| {
                            (
                                doc,
                                if doc % 17 == 0 {
                                    0
                                } else if doc > 8100 {
                                    1000
                                } else {
                                    tf
                                },
                            )
                        })
                        .collect()
                })
                .collect(),
        };
        let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let lists: Vec<_> = corpus
                .postings
                .iter()
                .enumerate()
                .map(|(i, postings)| {
                    let mut list = PostingList::new();
                    for &(doc, tf) in postings {
                        list.push(doc, tf);
                    }
                    (
                        BlockPostingList::from_posting_list_with_impact_bounds(
                            &list,
                            false,
                            Some(&|doc| lengths.length(doc).max(1)),
                            codec,
                        )
                        .unwrap(),
                        [1.2345, 0.0001, 9.876, 0.01][order[i]],
                    )
                })
                .collect();
            for params in [
                super::super::Bm25Params::default(),
                super::super::Bm25Params { k1: 0.0, b: 1.0 },
            ] {
                let truth = exhaustive(&corpus, &lists, true, 100.0, params);
                for required in 1..=lists.len() {
                    for k in [0, 1, 10, 1000] {
                        let mut expected: Vec<_> = truth
                            .iter()
                            .filter(|(doc, _)| {
                                **doc % 5 != 0
                                    && corpus.postings[..required].iter().all(|postings| {
                                        postings.binary_search_by_key(*doc, |&(id, _)| id).is_ok()
                                    })
                            })
                            .map(|(&doc, &score)| (doc, score.to_bits()))
                            .collect();
                        expected.sort_unstable_by(|a, b| {
                            f32::from_bits(b.1)
                                .total_cmp(&f32::from_bits(a.1))
                                .then(a.0.cmp(&b.0))
                        });
                        expected.truncate(k);
                        for seeded in [false, true] {
                            let mut executor = MaxScoreExecutor::text(
                                lists.clone(),
                                100.0,
                                k,
                                Some(&lengths),
                                params,
                                1.0,
                            )
                            .require_prefix_terms(required)
                            .with_predicate(Box::new(|doc| doc % 5 != 0));
                            if seeded
                                && let Some(&(_, bits)) = expected.last()
                                && f32::from_bits(bits).is_finite()
                            {
                                executor.seed_threshold(f32::from_bits(bits) * 0.9);
                            }
                            let got: Vec<_> = executor
                                .execute_sync()
                                .unwrap()
                                .into_iter()
                                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                                .collect();
                            assert_eq!(
                                got, expected,
                                "order={order:?} codec={codec:?} required={required} k={k} seeded={seeded} params={params:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    // A zero bound is not permission to omit required membership checks.
    // Terminal IDs and empty required/optional streams remain distinct.
    for empty_required in [false, true] {
        let mut must = PostingList::new();
        if !empty_required {
            for doc in [0, 2, 4097, u32::MAX - 1] {
                must.push(doc, 0);
            }
        }
        let mut opt = PostingList::new();
        for doc in [0, 1, 4097, u32::MAX - 1] {
            opt.push(doc, 1);
        }
        for empty_optional in [false, true] {
            let optional = if empty_optional {
                PostingList::new()
            } else {
                opt.clone()
            };
            let lists = vec![
                (BlockPostingList::from_posting_list(&must).unwrap(), 1.0),
                (BlockPostingList::from_posting_list(&optional).unwrap(), 2.0),
            ];
            let got = MaxScoreExecutor::text(
                lists,
                1.0,
                10,
                None,
                super::super::Bm25Params::default(),
                1.0,
            )
            .require_prefix_terms(1)
            .execute_sync()
            .unwrap();
            let mut docs: Vec<_> = got.iter().map(|hit| hit.doc_id).collect();
            docs.sort_unstable();
            assert_eq!(
                docs,
                if empty_required {
                    vec![]
                } else {
                    vec![0, 2, 4097, u32::MAX - 1]
                }
            );
        }
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Terms with very different densities (from 0.5% to 60% of the
/// documents), skewed term frequencies, and pseudo-random lengths.
fn random_corpus(seed: u64, n_docs: u32, n_terms: usize) -> Corpus {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let lengths: Vec<u16> = (0..n_docs)
        .map(|_| 1 + (xorshift(&mut state) % 400) as u16)
        .collect();
    let densities = [0.6, 0.25, 0.1, 0.03, 0.005];
    let postings = (0..n_terms)
        .map(|t| {
            let density = densities[t % densities.len()];
            let cutoff = (density * u32::MAX as f64) as u64;
            let mut postings = Vec::new();
            for doc in 0..n_docs {
                if (xorshift(&mut state) & 0xFFFF_FFFF) >= cutoff {
                    continue;
                }
                let r = xorshift(&mut state) % 100;
                let tf = if r < 70 {
                    1
                } else if r < 90 {
                    2
                } else {
                    3 + (r % 6) as u32
                };
                postings.push((doc, tf));
            }
            postings
        })
        .collect();
    Corpus {
        postings,
        lengths,
        n_docs,
    }
}

fn build_lists(
    corpus: &Corpus,
    lengths: Option<&crate::segment::chunk_map::DocLengths>,
) -> Vec<(crate::structures::BlockPostingList, f32)> {
    corpus
        .postings
        .iter()
        .map(|postings| {
            let mut list = crate::structures::PostingList::new();
            for &(doc, tf) in postings {
                list.push(doc, tf);
            }
            let length_of = lengths.map(|l| move |doc: DocId| l.length(doc));
            let block_list = crate::structures::BlockPostingList::from_posting_list_with(
                &list,
                false,
                length_of.as_ref().map(|f| f as &dyn Fn(DocId) -> u32),
            )
            .unwrap();
            let idf = super::super::bm25_idf(postings.len() as f32, corpus.n_docs as f32);
            (block_list, idf)
        })
        .collect()
}

/// Exhaustive per-document scores with the same formula the cursors use.
fn exhaustive(
    corpus: &Corpus,
    lists: &[(crate::structures::BlockPostingList, f32)],
    real_lengths: bool,
    avg: f32,
    params: super::super::Bm25Params,
) -> std::collections::HashMap<u32, f32> {
    let mut scores: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for (postings, (_, idf)) in corpus.postings.iter().zip(lists) {
        for &(doc, tf) in postings {
            let len = if real_lengths {
                corpus.lengths[doc as usize] as f32
            } else {
                tf as f32
            };
            *scores.entry(doc).or_insert(0.0) += params.score(tf as f32, *idf, len, avg);
        }
    }
    scores
}

fn check_top_k(
    label: &str,
    results: &[ScoredDoc],
    exhaustive: &std::collections::HashMap<u32, f32>,
    k: usize,
    predicate: Option<&dyn Fn(u32) -> bool>,
) {
    let mut expected: Vec<(u32, f32)> = exhaustive
        .iter()
        .filter(|(doc, _)| predicate.is_none_or(|p| p(**doc)))
        .map(|(doc, score)| (*doc, *score))
        .collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let want = k.min(expected.len());
    assert_eq!(results.len(), want, "{label}: result count");
    for (rank, (got, exp)) in results.iter().zip(&expected).enumerate() {
        let tolerance = 1e-4 * exp.1.abs().max(1.0);
        assert!(
            (got.score - exp.1).abs() <= tolerance,
            "{label}: rank {rank} score {} vs exhaustive {} (doc {} vs {})",
            got.score,
            exp.1,
            got.doc_id,
            exp.0
        );
        let own = exhaustive[&got.doc_id];
        assert!(
            (got.score - own).abs() <= tolerance,
            "{label}: doc {} scored {} but exhaustive says {}",
            got.doc_id,
            got.score,
            own
        );
        if let Some(p) = predicate {
            assert!(
                p(got.doc_id),
                "{label}: doc {} fails the predicate",
                got.doc_id
            );
        }
    }
    for pair in results.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "{label}: results not sorted"
        );
    }
}

/// The windowed executor returns the exact top-k (against an exhaustive
/// scorer and against the document-at-a-time loop) over corpora of
/// different sizes and term mixes, with and without real lengths,
/// predicates, and a seeded threshold.
#[test]
fn windowed_text_maxscore_matches_exhaustive_and_doc_at_a_time() {
    let params = super::super::Bm25Params::default();
    let predicate_fn = |doc: u32| !doc.is_multiple_of(3);
    let mut cases = 0usize;
    for seed in 1..=6u64 {
        for &n_docs in &[300u32, 2_500, 12_000] {
            for &n_terms in &[1usize, 2, 4, 9] {
                let corpus = random_corpus(seed, n_docs, n_terms);
                let doc_lengths =
                    crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
                for real_lengths in [true, false] {
                    let lengths = real_lengths.then_some(&doc_lengths);
                    let lists = build_lists(&corpus, lengths);
                    let avg = if real_lengths {
                        doc_lengths.avg_len()
                    } else {
                        1.0
                    };
                    let truth = exhaustive(&corpus, &lists, real_lengths, avg, params);
                    for &k in &[1usize, 10, 100] {
                        for with_predicate in [false, true] {
                            let label = format!(
                                "seed={seed} docs={n_docs} terms={n_terms} lengths={real_lengths} k={k} pred={with_predicate}"
                            );
                            let pred: Option<&dyn Fn(u32) -> bool> =
                                with_predicate.then_some(&predicate_fn);
                            let make = |seeded: f32| {
                                let mut executor = MaxScoreExecutor::text(
                                    lists.clone(),
                                    avg,
                                    k,
                                    lengths,
                                    params,
                                    1.0,
                                );
                                if with_predicate {
                                    executor = executor.with_predicate(Box::new(predicate_fn));
                                }
                                if seeded > 0.0 {
                                    executor.seed_threshold(seeded);
                                }
                                executor
                            };
                            let windowed = make(0.0).execute_windowed().unwrap();
                            check_top_k(&format!("windowed {label}"), &windowed, &truth, k, pred);
                            let reference = make(0.0).execute_doc_at_a_time_sync().unwrap();
                            check_top_k(&format!("reference {label}"), &reference, &truth, k, pred);
                            // A floor below the k-th score keeps the exact top-k.
                            if let Some(kth) = windowed.last().map(|r| r.score)
                                && windowed.len() == k
                            {
                                let seeded = make(kth * 0.9).execute_windowed().unwrap();
                                check_top_k(&format!("seeded {label}"), &seeded, &truth, k, pred);
                                // A floor above every score returns nothing.
                                let above =
                                    make(windowed[0].score * 1.5).execute_windowed().unwrap();
                                assert!(above.is_empty(), "{label}: floor above all scores");
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 400);
}

/// The approximate mode returns a subset of the exact top-k with exact
/// scores.
#[test]
fn windowed_text_maxscore_heap_factor_is_a_subset_with_exact_scores() {
    let params = super::super::Bm25Params::default();
    let corpus = random_corpus(7, 20_000, 6);
    let doc_lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
    let lists = build_lists(&corpus, Some(&doc_lengths));
    let avg = doc_lengths.avg_len();
    let truth = exhaustive(&corpus, &lists, true, avg, params);
    let exact = MaxScoreExecutor::text(lists.clone(), avg, 50, Some(&doc_lengths), params, 1.0)
        .execute_windowed()
        .unwrap();
    check_top_k("exact", &exact, &truth, 50, None);
    let approx = MaxScoreExecutor::text(lists, avg, 50, Some(&doc_lengths), params, 0.6)
        .execute_windowed()
        .unwrap();
    assert_eq!(approx.len(), 50);
    for hit in &approx {
        let own = truth[&hit.doc_id];
        assert!((hit.score - own).abs() <= 1e-4 * own.max(1.0));
    }
    // The usual heap-factor guarantee: every returned score is within the
    // factor of the exact k-th score, and the best document is exact.
    let exact_kth = exact.last().unwrap().score;
    assert!(approx.iter().all(|hit| hit.score >= exact_kth * 0.6 - 1e-4));
    assert_eq!(approx[0].doc_id, exact[0].doc_id);
    let overlap = approx
        .iter()
        .filter(|hit| exact.iter().any(|e| e.doc_id == hit.doc_id))
        .count();
    assert!(overlap >= 25, "overlap {overlap} of 50");
}

#[test]
fn text_heap_factor_below_one_actually_changes_pruning() {
    let params = super::super::Bm25Params::default();
    let corpus = random_corpus(7, 20_000, 6);
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
    let lists = build_lists(&corpus, Some(&lengths));
    let mut exact = MaxScoreExecutor::text(
        lists.clone(),
        lengths.avg_len(),
        50,
        Some(&lengths),
        params,
        1.0,
    );
    let mut approximate =
        MaxScoreExecutor::text(lists, lengths.avg_len(), 50, Some(&lengths), params, 0.01);
    let exact = exact.execute_windowed().unwrap();
    let approximate = approximate.execute_windowed().unwrap();
    assert_ne!(
        exact.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        approximate.iter().map(|hit| hit.doc_id).collect::<Vec<_>>()
    );
}

#[test]
fn text_and_sparse_factors_have_identical_conventions() {
    for factor in [0.0, 0.01, 0.5, 1.0] {
        let text = MaxScoreExecutor::text(
            Vec::new(),
            1.0,
            1,
            None,
            super::super::Bm25Params::default(),
            factor,
        );
        let generic = MaxScoreExecutor::new(Vec::new(), 1, factor);
        assert_eq!(text.inv_heap_factor, generic.inv_heap_factor);
        assert_eq!(text.inv_heap_factor, factor.clamp(0.01, 1.0).recip());
    }
}

#[test]
fn chunk_bounds_use_the_same_length_floor_as_scoring() {
    use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
    use crate::structures::{BlockPostingList, PostingList};
    let mut chunks = ChunkMapBuilder::default();
    let mut postings = PostingList::new();
    for id in 0..2048 {
        chunks
            .push(id / 4, (id % 4) as u16, if id % 4 == 0 { 1 } else { 200 })
            .unwrap();
        postings.push(id, 1);
    }
    let mut bytes = Vec::new();
    write_chunk_maps(&mut bytes, &[(0, &chunks)], &[]).unwrap();
    let maps = read_chunk_maps(crate::directories::OwnedBytes::new(bytes)).unwrap();
    let map = &maps.chunk_maps[&0];
    let list =
        BlockPostingList::from_posting_list_with(&postings, false, Some(&|id| map.length(id)))
            .unwrap();
    assert_eq!(list.min_len(), Some(1));
    let params = super::super::Bm25Params::default();
    let cursor = TermCursor::text_with_params(
        list,
        2.0,
        map.avg_len(),
        Some(LengthSource::Chunks(map)),
        params,
    );
    let expected = params.upper_bound_with_len(1.0, 2.0, 200.0, map.avg_len());
    assert_eq!(cursor.max_score.to_bits(), expected.to_bits());
    for block in 0..cursor.num_blocks {
        assert_eq!(cursor.text_block_bound(block).to_bits(), expected.to_bits());
        assert_eq!(
            cursor.text_group_bound(block).unwrap().to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn nested_scorers_keep_deadline_but_not_outer_score_floor() {
    let shared = SharedThreshold::for_limit(10).with_deadline(Some(std::time::Instant::now()));
    shared.raise(100.0);
    let options = super::super::ScorerOptions {
        shared_threshold: Some(shared.clone()),
        initial_threshold: 100.0,
        ..Default::default()
    }
    .without_threshold();
    assert_eq!(options.initial_threshold, 0.0);
    let nested = options.shared_threshold.unwrap();
    assert_eq!(nested.get(), 0.0);
    assert!(!nested.covers(10));
    assert!(nested.stop_if_expired());
    assert!(shared.truncated());
    nested.raise(200.0);
    assert_eq!(shared.get(), 100.0);
}

#[test]
fn score_bound_cache_reuses_only_its_last_key() {
    let cache = CachedScoreBound::new();
    assert_eq!(cache.get_or_compute(0, || 1.5), 1.5);
    assert_eq!(
        cache.get_or_compute(0, || panic!("recomputed active bound")),
        1.5
    );
    assert_eq!(cache.get_or_compute(1, || 2.5), 2.5);
    assert_eq!(cache.get_or_compute(0, || 3.5), 3.5);
}

#[test]
fn test_shared_threshold_monotonic_raise() {
    let shared = SharedThreshold::new();
    assert_eq!(shared.get(), 0.0);

    shared.raise(2.5);
    assert_eq!(shared.get(), 2.5);

    // Lower values never lower the floor.
    shared.raise(1.0);
    assert_eq!(shared.get(), 2.5);

    // Higher values raise it.
    shared.raise(4.0);
    assert_eq!(shared.get(), 4.0);

    // Non-positive and NaN are ignored.
    shared.raise(0.0);
    shared.raise(-3.0);
    shared.raise(f32::NAN);
    assert_eq!(shared.get(), 4.0);

    // Clones share the same atomic cell.
    let clone = shared.clone();
    clone.raise(9.0);
    assert_eq!(shared.get(), 9.0);
}

#[test]
fn test_shared_threshold_seed_matches_manual() {
    // A collector seeded with a floor prunes anything at/below it, matching
    // the threshold a fully-populated heap would have produced.
    let mut seeded = ScoreCollector::new(2);
    seeded.seed_threshold(3.0);
    assert_eq!(seeded.threshold(), 3.0);
    // A score at/below the floor cannot enter.
    assert!(!seeded.would_enter(3.0));
    assert!(seeded.would_enter(3.5));
    // Real inserts above the floor evict the sentinels; results contain no
    // sentinel (doc_id == u32::MAX) entries.
    seeded.insert(1, 5.0);
    seeded.insert(2, 4.0);
    let results = seeded.into_sorted_results();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert_eq!(results[1].0, 2);
}

#[test]
fn test_shared_threshold_can_raise_after_real_inserts() {
    let mut collector = ScoreCollector::new(3);
    collector.insert(1, 10.0);
    collector.insert(2, 4.0);
    assert_eq!(collector.real_len(), 2);

    // Raising the floor after traversal has started removes retained work
    // that can no longer reach the global top-k.
    collector.seed_threshold(6.0);
    assert_eq!(collector.threshold(), 6.0);
    assert_eq!(collector.real_len(), 1);

    // A real candidate tied with the floor displaces the sentinel because
    // its doc id wins the canonical tie break.
    assert!(collector.would_enter_candidate(3, 6.0, 0));
    assert!(collector.insert(3, 6.0));
    assert_eq!(collector.real_len(), 2);
    let results = collector.into_sorted_results();
    assert_eq!(results, vec![(1, 10.0, 0), (3, 6.0, 0)]);
}

#[test]
fn test_large_seed_uses_virtual_sentinels() {
    let k = 1_000_000_000;
    let mut collector = ScoreCollector::new(k);
    assert!(collector.heap.capacity() <= MAX_INITIAL_SCORE_COLLECTOR_CAPACITY);

    collector.seed_threshold(42.0);

    // Seeding a huge top-k is constant-time and does not materialize any
    // of its conceptual sentinel entries.
    assert_eq!(collector.heap.len(), 0);
    assert!(collector.heap.capacity() <= MAX_INITIAL_SCORE_COLLECTOR_CAPACITY);
    assert_eq!(collector.len(), k);
    assert_eq!(collector.real_len(), 0);
    assert_eq!(collector.threshold(), 42.0);
    assert!(!collector.is_empty());

    // A real result tied with the floor beats the sentinel by doc-id, while
    // a lower score remains below the conceptual threshold.
    assert!(collector.insert_with_ordinal(9, 42.0, 7));
    assert!(!collector.insert(10, 41.0));
    assert!(collector.insert(11, 43.0));
    assert_eq!(collector.len(), k);
    assert_eq!(collector.real_len(), 2);
    assert_eq!(
        collector.into_sorted_results(),
        vec![(11, 43.0, 0), (9, 42.0, 7)]
    );
}

#[test]
fn test_virtual_sentinels_preserve_tie_order_when_filled() {
    let mut collector = ScoreCollector::new(3);
    collector.seed_threshold(5.0);

    assert!(collector.insert_with_ordinal(3, 5.0, 2));
    assert!(collector.insert_with_ordinal(2, 5.0, 8));
    assert!(collector.insert_with_ordinal(1, 5.0, 4));
    assert_eq!(collector.real_len(), 3);
    assert!(collector.virtual_threshold.is_none());

    // Once all virtual slots have been displaced, canonical doc/ordinal
    // ordering still controls root replacement at an equal score.
    assert!(collector.insert_with_ordinal(2, 5.0, 1));
    assert!(!collector.insert_with_ordinal(4, 5.0, 0));
    assert_eq!(
        collector.into_sorted_results(),
        vec![(1, 5.0, 4), (2, 5.0, 1), (2, 5.0, 8)]
    );
}

#[test]
fn score_collection_preserves_total_order_ordinals_and_monotone_seeds() {
    use super::ScoreCollector;
    let compare = |a: &(u32, f32, u16), b: &(u32, f32, u16)| {
        b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)).then(a.2.cmp(&b.2))
    };
    for exceptional in [false, true] {
        let values = [
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.25,
            f32::INFINITY,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_0001),
        ];
        for k in [0, 1, 10, 257, 9000] {
            for seeded in [false, true] {
                let mut collector = ScoreCollector::new(k);
                let mut reference = Vec::new();
                let mut floor = None;
                for index in 0..8193u32 {
                    if seeded && [0, 2048, 4096].contains(&index) {
                        let seed = if index == 2048 { 7.5 } else { 1.25 };
                        collector.seed_threshold(seed);
                        floor = Some(floor.map_or(seed, |old: f32| old.max(seed)));
                        let sentinel = (u32::MAX, floor.unwrap(), 0);
                        reference.retain(|hit| compare(hit, &sentinel).is_lt());
                    }
                    // Later equal scores can win by document and ordinal.
                    let doc = 4096 - index / 2;
                    let ordinal = (index % 2) as u16;
                    let score = if exceptional && index % 17 == 0 {
                        values[(index / 17) as usize % values.len()]
                    } else {
                        ((index.wrapping_mul(15485863) % 1009) as f32 - 300.0) / 97.0
                    };
                    let hit = (doc, score, ordinal);
                    collector.insert_with_ordinal(doc, score, ordinal);
                    if floor.is_none_or(|floor| compare(&hit, &(u32::MAX, floor, 0)).is_lt()) {
                        reference.push(hit);
                    }
                }
                reference.sort_unstable_by(compare);
                reference.truncate(k);
                let bits = |hits: Vec<(u32, f32, u16)>| {
                    hits.into_iter()
                        .map(|(doc, score, ordinal)| (doc, score.to_bits(), ordinal))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(collector.into_sorted_results()),
                    bits(reference),
                    "exceptional={exceptional} k={k} seeded={seeded}"
                );
            }
        }
    }
}

#[test]
fn test_score_collector_basic() {
    let mut collector = ScoreCollector::new(3);

    collector.insert(1, 1.0);
    collector.insert(2, 2.0);
    collector.insert(3, 3.0);
    assert_eq!(collector.threshold(), 1.0);

    collector.insert(4, 4.0);
    assert_eq!(collector.threshold(), 2.0);

    let results = collector.into_sorted_results();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 4); // Highest score
    assert_eq!(results[1].0, 3);
    assert_eq!(results[2].0, 2);
}

#[test]
fn test_score_collector_threshold() {
    let mut collector = ScoreCollector::new(2);

    collector.insert(1, 5.0);
    collector.insert(2, 3.0);
    assert_eq!(collector.threshold(), 3.0);

    // Should not enter (score too low)
    assert!(!collector.would_enter(2.0));
    assert!(!collector.insert(3, 2.0));

    // Should enter (score high enough)
    assert!(collector.would_enter(4.0));
    assert!(collector.insert(4, 4.0));
    assert_eq!(collector.threshold(), 4.0);
}

#[test]
fn test_heap_entry_ordering() {
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        doc_id: 1,
        score: 3.0,
        ordinal: 0,
    });
    heap.push(HeapEntry {
        doc_id: 2,
        score: 1.0,
        ordinal: 0,
    });
    heap.push(HeapEntry {
        doc_id: 3,
        score: 2.0,
        ordinal: 0,
    });

    // Min-heap: lowest score should come out first
    assert_eq!(heap.pop().unwrap().score, 1.0);
    assert_eq!(heap.pop().unwrap().score, 2.0);
    assert_eq!(heap.pop().unwrap().score, 3.0);
}
#[test]
fn ratio_pruned_windows_match_exhaustive_score_bits_for_all_top_k_sizes() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    for terms in [1, 4] {
        let corpus = random_corpus(927, 20_000, terms);
        let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
        let lists: Vec<_> = corpus
            .postings
            .iter()
            .map(|postings| {
                let mut list = PostingList::new();
                for &(doc, tf) in postings {
                    list.push(doc, tf);
                }
                (
                    BlockPostingList::from_posting_list_with_ratio_bounds(
                        &list,
                        false,
                        Some(&|doc| lengths.length(doc)),
                        PostingCodec::Rounded,
                    )
                    .unwrap(),
                    super::super::bm25_idf(postings.len() as f32, corpus.n_docs as f32),
                )
            })
            .collect();
        for params in [
            super::super::Bm25Params::default(),
            super::super::Bm25Params { k1: 0.9, b: 0.0 },
            super::super::Bm25Params { k1: 4.0, b: 1.0 },
        ] {
            let scores = exhaustive(&corpus, &lists, true, lengths.avg_len(), params);
            let mut expected: Vec<_> = scores.into_iter().collect();
            expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for k in [10, 100, 1000] {
                let actual = MaxScoreExecutor::text(
                    lists.clone(),
                    lengths.avg_len(),
                    k,
                    Some(&lengths),
                    params,
                    1.0,
                )
                .execute_sync()
                .unwrap();
                assert_eq!(actual.len(), k);
                for (got, want) in actual.iter().zip(&expected) {
                    assert_eq!(
                        (got.doc_id, got.score.to_bits()),
                        (want.0, want.1.to_bits()),
                        "terms={terms} k={k} params={params:?}"
                    );
                }
            }
        }
    }
}
#[test]
fn single_ratio_blocks_keep_filtered_ties_and_stop_before_expired_work() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let mut postings = PostingList::new();
    for doc in 0..4096 {
        postings.push(doc, doc % 4 + 1);
    }
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&vec![100; 4096]);
    let list = BlockPostingList::from_posting_list_with_ratio_bounds(
        &postings,
        false,
        Some(&|_| 100),
        PostingCodec::Rounded,
    )
    .unwrap();
    let params = super::super::Bm25Params::default();
    let executor = || {
        MaxScoreExecutor::text(
            vec![(list.clone(), 2.0)],
            100.0,
            10,
            Some(&lengths),
            params,
            1.0,
        )
    };
    let results = executor()
        .with_predicate(Box::new(|doc| doc % 5 == 0))
        .execute_sync()
        .unwrap();
    assert_eq!(
        results.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
        (0..10).map(|i| 15 + 20 * i).collect::<Vec<_>>()
    );
    for hit in results {
        assert_eq!(
            hit.score.to_bits(),
            params.score(4.0, 2.0, 100.0, 100.0).to_bits()
        );
    }
    let budget = SharedThreshold::for_limit(10).with_deadline(Some(std::time::Instant::now()));
    let results = executor()
        .with_budget(Some(budget.clone()))
        .with_predicate(Box::new(|_| panic!("expired predicate was called")))
        .execute_sync()
        .unwrap();
    assert!(results.is_empty());
    assert!(budget.truncated());
}
// ── Review regressions ───────────────────────────────────────────────

/// A text block that fails to decode is index corruption, not the end of
/// the list: every execution path must surface it instead of returning
/// the postings decoded so far as a silently truncated top-k.
#[test]
fn corrupt_text_block_returns_error_instead_of_partial_results() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let params = super::super::Bm25Params::default();
    let mut postings = PostingList::new();
    for doc in 0..300u32 {
        postings.push(doc * 3, 1 + doc % 3);
    }
    let intact =
        BlockPostingList::from_posting_list_with_codec(&postings, PostingCodec::Packed).unwrap();
    assert_eq!(intact.num_blocks(), 3);

    let expect_corruption = |make: &dyn Fn() -> TermCursor<'static>, label: &str| {
        let mut cursor = make();
        assert!(
            matches!(
                cursor.ensure_block_loaded_sync(),
                Err(crate::Error::Corruption(_))
            ),
            "{label}: ensure_block_loaded_sync"
        );
        let mut cursor = make();
        assert!(
            matches!(cursor.advance_sync(), Err(crate::Error::Corruption(_))),
            "{label}: advance_sync"
        );
        let mut cursor = make();
        assert!(
            matches!(
                futures::executor::block_on(cursor.ensure_block_loaded()),
                Err(crate::Error::Corruption(_))
            ),
            "{label}: ensure_block_loaded"
        );
        let single = || MaxScoreExecutor::new(vec![make()], 10, 1.0);
        let pair = || MaxScoreExecutor::new(vec![make(), make()], 10, 1.0);
        let outcomes = [
            ("windowed", single().execute_sync()),
            ("async", futures::executor::block_on(single().execute())),
            ("doc-at-a-time", single().execute_doc_at_a_time_sync()),
            ("conjunction", pair().require_all_terms().execute_sync()),
            ("required", pair().require_prefix_terms(1).execute_sync()),
        ];
        for (path, outcome) in outcomes {
            assert!(
                matches!(outcome, Err(crate::Error::Corruption(_))),
                "{label}/{path}: {outcome:?}"
            );
        }
    };

    // Skip metadata claiming one block more than the stream holds: the
    // cursor sits on that phantom block and must not report exhaustion.
    let phantom = || {
        let mut cursor = TermCursor::text_with_params(intact.clone(), 1.0, 1.0, None, params);
        cursor.block_idx = cursor.num_blocks;
        cursor.num_blocks += 1;
        cursor
    };
    expect_corruption(&phantom, "phantom block");

    // A payload-only flip preserves structural admission but changes the
    // decoded endpoint; every executor must reject it at block load.
    let mut bytes = Vec::new();
    intact.serialize(&mut bytes).unwrap();
    bytes[8] ^= 0x01;
    let flipped = BlockPostingList::deserialize(&bytes).unwrap();
    expect_corruption(
        &|| TermCursor::text_with_params(flipped.clone(), 1.0, 1.0, None, params),
        "flipped payload",
    );
}

/// More cursors than the u64 cursor mask can track are dropped loudly
/// (weakest bounds first, counted), plain unions keep running over the
/// rest, and the `require_*` builders — whose semantics depend on the
/// input order the truncation destroyed — fail the query instead of
/// mis-ranking. Misuse without truncation is refused the same way.
#[test]
fn cursor_truncation_past_max_query_terms_is_observable_and_refuses_required_terms() {
    use crate::structures::{BlockPostingList, PostingList};
    let mut postings = PostingList::new();
    for doc in 0..64u32 {
        postings.push(doc, 1);
    }
    let list = BlockPostingList::from_posting_list(&postings).unwrap();
    let params = super::super::Bm25Params { k1: 0.0, b: 1.0 };
    let cursors = |count: usize| -> Vec<TermCursor<'static>> {
        (0..count)
            .map(|i| TermCursor::text_with_params(list.clone(), 1.0 + i as f32, 1.0, None, params))
            .collect()
    };
    let limit = super::super::MAX_QUERY_TERMS;
    let extra = 3;
    let executor = MaxScoreExecutor::new(cursors(limit + extra), 10, 1.0);
    assert_eq!(executor.cursors.len(), limit);
    assert_eq!(executor.dropped_cursors, extra);
    assert!(
        executor
            .cursors
            .iter()
            .all(|cursor| cursor.max_score >= 1.0 + extra as f32),
        "the weakest cursors are the ones dropped"
    );
    let hits = executor.execute_sync().unwrap();
    assert_eq!(hits.len(), 10);
    let expected: f32 = (extra..limit + extra).map(|i| 1.0 + i as f32).sum();
    assert!((hits[0].score - expected).abs() < 1e-3);

    let is_query_error =
        |outcome: crate::Result<Vec<ScoredDoc>>| matches!(outcome, Err(crate::Error::Query(_)));
    assert!(is_query_error(
        MaxScoreExecutor::new(cursors(limit + 1), 10, 1.0)
            .require_prefix_terms(1)
            .execute_sync()
    ));
    assert!(is_query_error(
        MaxScoreExecutor::new(cursors(limit + 1), 10, 1.0)
            .require_all_terms()
            .execute_sync()
    ));
    assert!(is_query_error(futures::executor::block_on(
        MaxScoreExecutor::new(cursors(limit + 1), 10, 1.0)
            .require_prefix_terms(limit)
            .execute()
    )));
    assert!(is_query_error(
        MaxScoreExecutor::new(cursors(1), 10, 1.0)
            .require_all_terms()
            .execute_sync()
    ));
    assert!(is_query_error(
        MaxScoreExecutor::new(cursors(2), 10, 1.0)
            .require_prefix_terms(0)
            .execute_sync()
    ));
    assert!(is_query_error(
        MaxScoreExecutor::new(cursors(2), 10, 1.0)
            .require_prefix_terms(3)
            .execute_sync()
    ));
    // Valid configurations at the limit are untouched.
    assert_eq!(
        MaxScoreExecutor::new(cursors(limit), 10, 1.0)
            .require_prefix_terms(limit)
            .execute_sync()
            .unwrap()
            .len(),
        10
    );
}

/// Bound sums accumulate in ascending-bound order while the
/// document-at-a-time loop credits a candidate essential-first, so
/// above 16 (one ulp = 1.9e-6) the two differ by several ulps. The loop
/// must prune with the same relative margin as every other path: a
/// candidate whose bound sum lands three ulps under the floor but whose
/// credited score clears it is scored, not dropped (the old absolute
/// `1e-6` margin — half an ulp here — pruned it).
#[test]
fn doc_at_a_time_bounds_within_rounding_of_the_threshold_do_not_prune_acceptable_candidates() {
    use crate::structures::{BlockPostingList, PostingList};
    let ulp = 16f32.next_up() - 16.0;
    let mut single = PostingList::new();
    single.push(0, 1);
    let list = BlockPostingList::from_posting_list(&single).unwrap();
    // `k1 = 0`: every score equals its idf, which is also its bound.
    let params = super::super::Bm25Params { k1: 0.0, b: 1.0 };
    // One strong term plus 18 weak ones just above half an ulp of it:
    // the ascending prefix sums them exactly (16 + 13.5 ulp → 16 + 14),
    // crediting them one by one on top of 16 rounds each up to a whole
    // ulp (16 + 18).
    let mut lists = vec![(list.clone(), 16.0)];
    lists.extend((0..18).map(|_| (list.clone(), 0.75 * ulp)));
    let floor = 16.0 + 17.0 * ulp;
    let mut executor = MaxScoreExecutor::text(lists, 1.0, 1, None, params, 1.0);
    executor.seed_threshold(floor);
    let hits = executor.execute_doc_at_a_time_sync().unwrap();
    assert_eq!(hits.len(), 1, "candidate pruned by bound rounding");
    assert_eq!(hits[0].doc_id, 0);
    assert!(hits[0].score > floor, "{} vs floor {floor}", hits[0].score);
}

/// The conjunction path honours the query deadline: expired before any
/// work it scores nothing, expired mid-run it stops within its check
/// interval; both flag the result truncated and every returned hit is
/// exact.
#[test]
fn conjunction_stops_at_the_deadline_and_flags_truncation() {
    use crate::structures::{BlockPostingList, PostingList};
    let n_docs = 20_000u32;
    let lists: Vec<_> = [2usize, 3]
        .iter()
        .map(|&step| {
            let mut postings = PostingList::new();
            for doc in (0..n_docs).step_by(step) {
                postings.push(doc, 1 + doc % 4);
            }
            (BlockPostingList::from_posting_list(&postings).unwrap(), 1.5)
        })
        .collect();
    let params = super::super::Bm25Params::default();
    let make = |budget: Option<SharedThreshold>| {
        MaxScoreExecutor::text(lists.clone(), 2.0, 100, None, params, 1.0)
            .require_all_terms()
            .with_budget(budget)
    };
    let full = make(None).execute_sync().unwrap();
    assert_eq!(full.len(), 100);

    let expired = SharedThreshold::for_limit(100).with_deadline(Some(std::time::Instant::now()));
    let hits = make(Some(expired.clone()))
        .with_predicate(Box::new(|_| {
            panic!("expired conjunction called its predicate")
        }))
        .execute_sync()
        .unwrap();
    assert!(hits.is_empty());
    assert!(expired.truncated());

    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5);
    let budget = SharedThreshold::for_limit(100).with_deadline(Some(deadline));
    let first = std::sync::atomic::AtomicBool::new(true);
    let hits = make(Some(budget.clone()))
        .with_predicate(Box::new(move |_| {
            if first.swap(false, std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            true
        }))
        .execute_sync()
        .unwrap();
    assert!(budget.truncated());
    assert!(
        hits.len() < full.len(),
        "{} hits after the deadline",
        hits.len()
    );
    for hit in &hits {
        assert!(full.iter().any(
            |exact| exact.doc_id == hit.doc_id && exact.score.to_bits() == hit.score.to_bits()
        ));
    }
}

/// `text_chunked` scores virtual chunk ids with their real chunk lengths
/// (`LengthSource::Chunks`) end to end through the windowed executor,
/// bit-identical to the exhaustive scorer, for one and several terms.
#[test]
fn chunked_lengths_score_windows_like_the_exhaustive_scorer() {
    use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let n_chunks = 6000u32;
    let mut state = 0x1234_5678_9abc_def1u64;
    let mut chunks = ChunkMapBuilder::default();
    for vid in 0..n_chunks {
        let tokens = 1 + (xorshift(&mut state) % 300) as u32;
        chunks.push(vid / 3, (vid % 3) as u16, tokens).unwrap();
    }
    let mut bytes = Vec::new();
    write_chunk_maps(&mut bytes, &[(0, &chunks)], &[]).unwrap();
    let maps = read_chunk_maps(crate::directories::OwnedBytes::new(bytes)).unwrap();
    let map = &maps.chunk_maps[&0];
    let terms: Vec<Vec<(u32, u32)>> = [(2u32, 0u32), (5, 1), (11, 3)]
        .iter()
        .map(|&(modulus, residue)| {
            (0..n_chunks)
                .filter(|vid| vid % modulus == residue)
                .map(|vid| (vid, 1 + vid % 3))
                .collect()
        })
        .collect();
    let params = super::super::Bm25Params::default();
    let avg = map.avg_len();
    let all_lists: Vec<_> = terms
        .iter()
        .map(|postings| {
            let mut list = PostingList::new();
            for &(vid, tf) in postings {
                list.push(vid, tf);
            }
            (
                BlockPostingList::from_posting_list_with_impact_bounds(
                    &list,
                    false,
                    Some(&|vid| map.length(vid)),
                    PostingCodec::Rounded,
                )
                .unwrap(),
                super::super::bm25_idf(postings.len() as f32, n_chunks as f32),
            )
        })
        .collect();
    for width in [1, 3] {
        let lists = all_lists[..width].to_vec();
        let mut truth = std::collections::HashMap::<u32, f32>::new();
        for (postings, (_, idf)) in terms.iter().zip(&lists) {
            for &(vid, tf) in postings {
                *truth.entry(vid).or_insert(0.0) +=
                    params.score(tf as f32, *idf, map.bm25_length(vid) as f32, avg);
            }
        }
        let mut expected: Vec<_> = truth.into_iter().collect();
        expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for k in [1, 10, 200] {
            let hits = MaxScoreExecutor::text_chunked(lists.clone(), avg, k, map, params, 1.0)
                .execute_sync()
                .unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| (hit.doc_id, hit.score.to_bits()))
                    .collect::<Vec<_>>(),
                expected[..k]
                    .iter()
                    .map(|&(vid, score)| (vid, score.to_bits()))
                    .collect::<Vec<_>>(),
                "terms={width} k={k}"
            );
        }
    }
}

/// With impact bounds, whole L1 groups of long (low-scoring) documents
/// are skipped once the heap holds short ones, and the result is still
/// bit-identical to the exhaustive scorer.
#[test]
fn impact_bound_groups_are_skipped_while_matching_the_exhaustive_scorer() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let n_docs = 65_536u32;
    let lengths: Vec<u16> = (0..n_docs)
        .map(|doc| if doc < 2048 { 20 } else { 400 })
        .collect();
    let doc_lengths = crate::segment::chunk_map::DocLengths::from_lengths(&lengths);
    let corpus = Corpus {
        postings: vec![
            (0..n_docs)
                .filter(|doc| doc % 3 == 0)
                .map(|doc| (doc, 1 + doc % 2))
                .collect(),
            (0..n_docs)
                .filter(|doc| doc % 7 == 0)
                .map(|doc| (doc, 1))
                .collect(),
        ],
        lengths,
        n_docs,
    };
    let lists: Vec<_> = corpus
        .postings
        .iter()
        .map(|postings| {
            let mut list = PostingList::new();
            for &(doc, tf) in postings {
                list.push(doc, tf);
            }
            (
                BlockPostingList::from_posting_list_with_impact_bounds(
                    &list,
                    false,
                    Some(&|doc| doc_lengths.length(doc)),
                    PostingCodec::Rounded,
                )
                .unwrap(),
                super::super::bm25_idf(postings.len() as f32, n_docs as f32),
            )
        })
        .collect();
    assert!(lists.iter().all(|(list, _)| list.has_group_impact_bounds()));
    let params = super::super::Bm25Params::default();
    let avg = doc_lengths.avg_len();
    let mut expected: Vec<_> = exhaustive(&corpus, &lists, true, avg, params)
        .into_iter()
        .collect();
    expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut executor =
        MaxScoreExecutor::text(lists.clone(), avg, 10, Some(&doc_lengths), params, 1.0);
    let hits = executor.execute_windowed().unwrap();
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        expected[..10]
            .iter()
            .map(|&(doc, score)| (doc, score.to_bits()))
            .collect::<Vec<_>>()
    );
    assert!(executor.stats.groups_skipped > 0, "{:?}", executor.stats);
}

/// `heap_factor < 1` through the single ratio-bounded cursor path: the
/// approximate run decodes fewer blocks, and every returned score is the
/// exact score of its document within the factor of the exact k-th.
#[test]
fn single_text_heap_factor_below_one_prunes_more_blocks_with_exact_scores() {
    use crate::structures::{BlockPostingList, PostingCodec, PostingList};
    let corpus = random_corpus(11, 40_000, 1);
    let lengths = crate::segment::chunk_map::DocLengths::from_lengths(&corpus.lengths);
    let mut list = PostingList::new();
    for &(doc, tf) in &corpus.postings[0] {
        list.push(doc, tf);
    }
    let list = BlockPostingList::from_posting_list_with_ratio_bounds(
        &list,
        false,
        Some(&|doc| lengths.length(doc)),
        PostingCodec::Rounded,
    )
    .unwrap();
    let idf = super::super::bm25_idf(corpus.postings[0].len() as f32, corpus.n_docs as f32);
    let lists = vec![(list, idf)];
    let params = super::super::Bm25Params::default();
    let avg = lengths.avg_len();
    let truth = exhaustive(&corpus, &lists, true, avg, params);
    let mut expected: Vec<_> = truth.iter().map(|(&doc, &score)| (doc, score)).collect();
    expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let make =
        |factor| MaxScoreExecutor::text(lists.clone(), avg, 50, Some(&lengths), params, factor);

    let mut exact = make(1.0);
    assert!(exact.single_text_with_ratio_bounds());
    let exact_hits = exact.execute_single_text().unwrap();
    assert_eq!(
        exact_hits
            .iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect::<Vec<_>>(),
        expected[..50]
            .iter()
            .map(|&(doc, score)| (doc, score.to_bits()))
            .collect::<Vec<_>>()
    );

    let mut approximate = make(0.5);
    let approximate_hits = approximate.execute_single_text().unwrap();
    assert_eq!(approximate_hits.len(), 50);
    let exact_kth = exact_hits.last().unwrap().score;
    for hit in &approximate_hits {
        assert_eq!(hit.score.to_bits(), truth[&hit.doc_id].to_bits());
        assert!(hit.score >= exact_kth * 0.5 - 1e-4);
    }
    assert!(
        approximate.stats.blocks_scored < exact.stats.blocks_scored,
        "approximate {:?} vs exact {:?}",
        approximate.stats,
        exact.stats
    );
}

/// Once the heap fills with documents that need the rare optional term,
/// the required-prefix windows switch to that cheaper term as the lead;
/// the switch is counted and the result is still the exhaustive top-k
/// over documents matching the required term.
#[test]
fn required_windows_switch_to_a_cheaper_optional_lead_and_match_exhaustive() {
    use crate::structures::{BlockPostingList, PostingList};
    let n_docs = 32_768u32;
    let corpus = Corpus {
        postings: vec![
            (0..n_docs).map(|doc| (doc, 1)).collect(),
            (0..n_docs)
                .filter(|doc| doc % 1000 == 0)
                .map(|doc| (doc, 1))
                .collect(),
        ],
        lengths: vec![1; n_docs as usize],
        n_docs,
    };
    let lists: Vec<_> = corpus
        .postings
        .iter()
        .zip([0.1f32, 10.0])
        .map(|(postings, idf)| {
            let mut list = PostingList::new();
            for &(doc, tf) in postings {
                list.push(doc, tf);
            }
            (BlockPostingList::from_posting_list(&list).unwrap(), idf)
        })
        .collect();
    let params = super::super::Bm25Params { k1: 0.0, b: 1.0 };
    let mut expected: Vec<_> = exhaustive(&corpus, &lists, false, 1.0, params)
        .into_iter()
        .collect();
    expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let bits = |hits: &[ScoredDoc]| {
        hits.iter()
            .map(|hit| (hit.doc_id, hit.score.to_bits()))
            .collect::<Vec<_>>()
    };
    let wanted: Vec<_> = expected[..5]
        .iter()
        .map(|&(doc, score)| (doc, score.to_bits()))
        .collect();
    let mut executor =
        MaxScoreExecutor::text(lists.clone(), 1.0, 5, None, params, 1.0).require_prefix_terms(1);
    let hits = executor.execute_text_windows::<true>().unwrap();
    assert_eq!(bits(&hits), wanted);
    assert!(executor.stats.optional_leads > 0, "{:?}", executor.stats);
    assert!(executor.stats.optional_leads < executor.stats.windows);
    let through_dispatch = MaxScoreExecutor::text(lists, 1.0, 5, None, params, 1.0)
        .require_prefix_terms(1)
        .execute_sync()
        .unwrap();
    assert_eq!(bits(&through_dispatch), wanted);
}

/// The per-thread window scratch outlives a query: a narrower query run
/// after a wider one on the same thread must see none of the wider
/// query's contributions (stale slots are masked by presence bits).
#[test]
fn window_scratch_reuse_across_queries_keeps_scores_bit_identical() {
    let params = super::super::Bm25Params::default();
    let wide = random_corpus(5, 12_000, 6);
    let narrow = random_corpus(9, 3_000, 2);
    let wide_lengths = crate::segment::chunk_map::DocLengths::from_lengths(&wide.lengths);
    let narrow_lengths = crate::segment::chunk_map::DocLengths::from_lengths(&narrow.lengths);
    let wide_lists = build_lists(&wide, Some(&wide_lengths));
    let narrow_lists = build_lists(&narrow, Some(&narrow_lengths));
    let run_narrow = || {
        MaxScoreExecutor::text(
            narrow_lists.clone(),
            narrow_lengths.avg_len(),
            100,
            Some(&narrow_lengths),
            params,
            1.0,
        )
        .execute_sync()
        .unwrap()
        .into_iter()
        .map(|hit| (hit.doc_id, hit.score.to_bits()))
        .collect::<Vec<_>>()
    };
    let fresh = std::thread::scope(|scope| scope.spawn(run_narrow).join().unwrap());
    assert!(
        !MaxScoreExecutor::text(
            wide_lists,
            wide_lengths.avg_len(),
            100,
            Some(&wide_lengths),
            params,
            1.0
        )
        .execute_sync()
        .unwrap()
        .is_empty()
    );
    assert_eq!(run_narrow(), fresh);
    assert_eq!(run_narrow(), fresh);
}
#[test]
fn mapped_batch_admission_resolves_only_competitive_scores_and_keeps_stable_ties() {
    use std::cell::Cell;
    let docs: Vec<_> = (0..257).collect();
    let mut scores = vec![1.0; docs.len()];
    scores[0] = 100.0;
    scores[256] = 100.0;
    for k in [1, 10, 1000] {
        for seed in [0.0, 50.0] {
            let calls = Cell::new(0);
            let resolve = |doc| {
                calls.set(calls.get() + 1);
                256 - doc
            };
            let mut actual = ScoreCollector::new(k);
            actual.seed_threshold(seed);
            actual.insert_text_run_with_mapping(&docs, &scores, resolve);
            let mut expected = ScoreCollector::new(k);
            expected.seed_threshold(seed);
            for (&doc, &score) in docs.iter().zip(&scores) {
                expected.insert(256 - doc, score);
            }
            let bits = |collector: ScoreCollector| {
                collector
                    .into_sorted_results()
                    .into_iter()
                    .map(|hit| (hit.0, hit.1.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(actual), bits(expected));
            if k == 1 {
                assert!(calls.get() <= 9, "resolved {} IDs", calls.get());
            }
        }
    }
}

#[test]
fn batched_text_admission_preserves_total_order_and_seeded_heaps() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc00001),
        f32::from_bits(0xffc00001),
        1.0,
    ];
    for count in [0, 1, 7, 8, 9, 63, 128, 257] {
        let docs: Vec<_> = (0..count as u32).rev().collect();
        let scores: Vec<_> = (0..count).map(|i| values[i % values.len()]).collect();
        for k in [0, 1, 7, 8, 10, 64, 300] {
            for seed in [0.0, 1.0, 3.0] {
                let mut scalar = ScoreCollector::new(k);
                let mut batch = ScoreCollector::new(k);
                scalar.seed_threshold(seed);
                batch.seed_threshold(seed);
                for (&doc, &score) in docs.iter().zip(&scores) {
                    scalar.insert(doc, 0.0 + score);
                }
                batch.insert_text_run(&docs, &scores);
                let bits = |c: ScoreCollector| {
                    c.into_sorted_results()
                        .into_iter()
                        .map(|(d, s, o)| (d, s.to_bits(), o))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(batch),
                    bits(scalar),
                    "count={count}, k={k}, seed={seed}"
                );
            }
        }
    }
}

#[cfg(feature = "query-diagnostics")]
#[test]
fn diagnostic_heap_updates_count_admissions_and_exclude_rejections() {
    let ((hits, empty), work) = crate::search_diagnostics::capture_sync(|| {
        let mut heap = super::ScoreCollector::new(2);
        assert!(heap.insert(7, 1.0));
        assert!(heap.insert(8, 2.0));
        assert!(!heap.insert(9, 0.5));
        assert!(heap.insert(6, 1.0)); // Equal-score stable-ID replacement.
        assert!(!heap.insert(10, 1.0));
        assert!(heap.insert(5, 3.0));
        let mut empty = super::ScoreCollector::new(0);
        assert!(!empty.insert(0, 100.0));
        (heap.len(), empty.len())
    });
    assert_eq!((hits, empty), (2, 0));
    assert_eq!(work.maxscore_heap_updates, 4);
}
