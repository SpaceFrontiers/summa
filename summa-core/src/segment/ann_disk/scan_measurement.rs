//! Reproducible persisted binary-leaf scan measurement. Kept ignored because
//! its latency samples require an otherwise idle release-build host.
use super::*;

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[test]
#[ignore = "release benchmark; run alone with --nocapture"]
fn measure_persisted_binary_leaf_scans() {
    const ROWS: usize = 131_072;
    const LEAVES: usize = 32;
    const QUERIES: usize = 16;
    for width in [32, 320] {
        let mut seed = 42;
        let codes: Vec<u8> = (0..ROWS * width).map(|_| random(&mut seed) as u8).collect();
        let queries: Vec<Vec<u8>> = (0..QUERIES)
            .map(|_| (0..width).map(|_| random(&mut seed) as u8).collect())
            .collect();
        let docs: Vec<u32> = (0..ROWS as u32).collect();
        let ordinals = vec![0; ROWS];
        let runs: Vec<_> = (0..LEAVES)
            .map(|leaf| {
                let start = leaf * ROWS / LEAVES;
                let end = (leaf + 1) * ROWS / LEAVES;
                BuildRun {
                    cluster_id: leaf as u32,
                    doc_ids: &docs[start..end],
                    ordinals: &ordinals[start..end],
                    codes: &codes[start * width..end * width],
                }
            })
            .collect();
        let header = AnnDiskHeader {
            kind: AnnKind::BinaryIvf,
            routing: IvfRoutingMode::Flat,
            dim: width * 8,
            code_size: width,
            num_clusters: LEAVES as u32,
            quantizer_version: 42,
            codebook_version: 0,
            vector_count: ROWS,
        };
        let mut bytes = Vec::new();
        write_built_runs(header, &runs, &mut bytes, None).unwrap();
        // Hash the persisted bytes as well as each result for cross-build audits.
        let byte_hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        let disk =
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, ROWS as u32).unwrap();
        for probes in [1, 8, 32] {
            let leaves: Vec<u32> = (0..probes).collect();
            for k in [10, 100] {
                let mut hash = 0xcbf29ce484222325u64;
                for query in &queries {
                    let actual = disk
                        .search_binary_clusters_with_tuning::<false>(query, k, &leaves, usize::MAX)
                        .unwrap();
                    // Independent exact Hamming oracle for the same probed rows.
                    let inv_dim = 1.0 / (width * 8) as f32;
                    let mut oracle: Vec<_> = codes[..probes as usize * ROWS / LEAVES * width]
                        .chunks_exact(width)
                        .enumerate()
                        .map(|(id, code)| {
                            let distance: u32 = query
                                .iter()
                                .zip(code)
                                .map(|(&a, &b)| (a ^ b).count_ones())
                                .sum();
                            (id as u32, 0u16, 1.0 - distance as f32 * inv_dim)
                        })
                        .collect();
                    oracle.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
                    oracle.truncate(k);
                    assert_eq!(actual, oracle);
                    for (id, ordinal, score) in actual {
                        for value in [id, u32::from(ordinal), score.to_bits()] {
                            hash = (hash ^ u64::from(value)).wrapping_mul(0x100000001b3);
                        }
                    }
                }
                let mut samples = Vec::new();
                for _ in 0..7 {
                    for query in &queries {
                        let start = std::time::Instant::now();
                        std::hint::black_box(
                            disk.search_binary_clusters_with_tuning::<false>(
                                std::hint::black_box(query),
                                k,
                                &leaves,
                                usize::MAX,
                            )
                            .unwrap(),
                        );
                        samples.push(start.elapsed().as_nanos() as u64);
                    }
                }
                println!(
                    "BINARY_SCAN {{\"width\":{width},\"rows\":{ROWS},\"probes\":{probes},\"k\":{k},\"bytes_hash\":\"{byte_hash:016x}\",\"result_hash\":\"{hash:016x}\",\"samples_ns\":{samples:?}}}"
                );
            }
        }
    }
}
