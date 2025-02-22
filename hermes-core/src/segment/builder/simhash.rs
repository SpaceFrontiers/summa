/// SplitMix64 finalizer — good avalanche for 32→64 bit expansion.
/// Used to derive SimHash bit patterns from dimension IDs.
///
/// Based on the standard splitmix64 output function (Steele et al.).
/// Starts from `dim_id + 1` to avoid the fixed point at zero.
#[inline]
pub(crate) fn stafford_mix(dim_id: u32) -> u64 {
    let mut h = dim_id as u64 + 1; // +1 avoids fixed point: stafford_mix(0) != 0
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb); // odd multiplier — all 64 output bits active
    h ^= h >> 31;
    h
}

/// Accumulate one (dim_id, impact) pair into a 64-element SimHash accumulator.
///
/// This is the shared primitive used by both `simhash_from_sparse_vector`
/// (build-time) and `reorder_bmp_blob` (reorder-time).
#[inline]
pub(crate) fn simhash_accumulate(acc: &mut [i32; 64], dim_id: u32, impact: i32) {
    let mut mask = stafford_mix(dim_id);
    for chunk in acc.chunks_exact_mut(4) {
        chunk[0] += if mask & 1 != 0 { impact } else { -impact };
        chunk[1] += if mask & 2 != 0 { impact } else { -impact };
        chunk[2] += if mask & 4 != 0 { impact } else { -impact };
        chunk[3] += if mask & 8 != 0 { impact } else { -impact };
        mask >>= 4;
    }
}

/// Convert a 64-element accumulator to a SimHash fingerprint.
/// Bit i = 1 iff accumulator[i] > 0.
#[inline]
pub(crate) fn simhash_finalize(acc: &[i32; 64]) -> u64 {
    let mut hash = 0u64;
    for (bit, &a) in acc.iter().enumerate() {
        if a > 0 {
            hash |= 1u64 << bit;
        }
    }
    hash
}

/// Compute SimHash from a sparse vector using quantized u8 impacts.
///
/// This matches exactly what `reorder_bmp_blob` computes from BMP block data,
/// ensuring build-time and reorder-time SimHash are identical. Entries below
/// `weight_threshold` or that quantize to zero impact are excluded — same as
/// what survives into the BMP block postings.
#[inline]
pub fn simhash_from_sparse_vector(
    entries: &[(u32, f32)],
    weight_threshold: f32,
    max_weight: f32,
) -> u64 {
    let mut acc = [0i32; 64];
    for &(dim, weight) in entries {
        let abs_w = weight.abs();
        if abs_w < weight_threshold {
            continue;
        }
        // Same quantization as build_bmp_blob / quantize_weight
        let impact = if max_weight <= 0.0 {
            0u8
        } else {
            (abs_w / max_weight * 255.0).round().clamp(0.0, 255.0) as u8
        };
        if impact == 0 {
            continue;
        }
        simhash_accumulate(&mut acc, dim, impact as i32);
    }
    simhash_finalize(&acc)
}

/// Compute per-bit majority vote SimHash from a set of hashes.
///
/// For each of the 64 bit positions, sets the output bit to 1 if more than
/// half the input hashes have that bit set. This produces the "centroid"
/// in SimHash space — the representative that minimizes Hamming distance
/// to the input set.
pub fn majority_simhash(hashes: &[u64]) -> u64 {
    if hashes.is_empty() {
        return 0;
    }
    let threshold = hashes.len() / 2;
    let mut result: u64 = 0;
    for bit in 0..64 {
        let count = hashes.iter().filter(|&&h| h & (1u64 << bit) != 0).count();
        if count > threshold {
            result |= 1u64 << bit;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hamming_distance(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    /// Identical vectors produce the same hash.
    #[test]
    fn test_simhash_identical() {
        let v = vec![(10, 1.0), (20, 0.5), (100, 2.0)];
        let h1 = simhash_from_sparse_vector(&v, 0.0, 5.0);
        let h2 = simhash_from_sparse_vector(&v, 0.0, 5.0);
        assert_eq!(h1, h2);
    }

    /// Similar vectors (large overlap) produce close hashes (small Hamming distance).
    #[test]
    fn test_simhash_similar_vectors_close() {
        // Base vector: dims 100..200 with weight 1.0
        let base: Vec<(u32, f32)> = (100..200).map(|d| (d, 1.0)).collect();

        // Similar: 90% overlap (dims 100..190 shared, 10 new dims)
        let mut similar = base[..90].to_vec();
        similar.extend((200..210).map(|d| (d, 1.0)));

        // Dissimilar: completely different dims
        let dissimilar: Vec<(u32, f32)> = (5000..5100).map(|d| (d, 1.0)).collect();

        let h_base = simhash_from_sparse_vector(&base, 0.0, 5.0);
        let h_similar = simhash_from_sparse_vector(&similar, 0.0, 5.0);
        let h_dissimilar = simhash_from_sparse_vector(&dissimilar, 0.0, 5.0);

        let dist_similar = hamming_distance(h_base, h_similar);
        let dist_dissimilar = hamming_distance(h_base, h_dissimilar);

        assert!(
            dist_similar < dist_dissimilar,
            "Similar vectors should have smaller Hamming distance: similar={}, dissimilar={}",
            dist_similar,
            dist_dissimilar,
        );
        // 90% overlap should yield very close hashes (< 16 out of 64 bits differ)
        assert!(
            dist_similar <= 16,
            "90% overlap should produce Hamming distance ≤ 16, got {}",
            dist_similar,
        );
    }

    /// Vectors from the same "topic" (shared dim range) cluster together.
    #[test]
    fn test_simhash_topic_clustering() {
        let mut rng_seed = 42u64;
        let mut next = || -> u32 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_seed >> 33) as u32
        };

        // Topic A: dims 0..500
        let make_topic_a = |next: &mut dyn FnMut() -> u32| -> Vec<(u32, f32)> {
            let mut v: Vec<(u32, f32)> = (0..80)
                .map(|_| {
                    let d = next() % 500;
                    (d, 0.5 + (next() % 100) as f32 / 100.0)
                })
                .collect();
            v.sort_by_key(|&(d, _)| d);
            v.dedup_by_key(|e| e.0);
            v
        };

        // Topic B: dims 10000..10500
        let make_topic_b = |next: &mut dyn FnMut() -> u32| -> Vec<(u32, f32)> {
            let mut v: Vec<(u32, f32)> = (0..80)
                .map(|_| {
                    let d = 10000 + next() % 500;
                    (d, 0.5 + (next() % 100) as f32 / 100.0)
                })
                .collect();
            v.sort_by_key(|&(d, _)| d);
            v.dedup_by_key(|e| e.0);
            v
        };

        // Generate 10 vectors from each topic
        let topic_a: Vec<u64> = (0..10)
            .map(|_| simhash_from_sparse_vector(&make_topic_a(&mut next), 0.0, 5.0))
            .collect();
        let topic_b: Vec<u64> = (0..10)
            .map(|_| simhash_from_sparse_vector(&make_topic_b(&mut next), 0.0, 5.0))
            .collect();

        // Intra-topic distances should be small
        let mut intra_dists = Vec::new();
        for i in 0..topic_a.len() {
            for j in (i + 1)..topic_a.len() {
                intra_dists.push(hamming_distance(topic_a[i], topic_a[j]));
            }
            for j in (i + 1)..topic_b.len() {
                intra_dists.push(hamming_distance(topic_b[i], topic_b[j]));
            }
        }

        // Inter-topic distances should be large
        let mut inter_dists = Vec::new();
        for &ha in &topic_a {
            for &hb in &topic_b {
                inter_dists.push(hamming_distance(ha, hb));
            }
        }

        let avg_intra = intra_dists.iter().sum::<u32>() as f64 / intra_dists.len() as f64;
        let avg_inter = inter_dists.iter().sum::<u32>() as f64 / inter_dists.len() as f64;

        // Key property: same-topic vectors are closer than cross-topic vectors
        assert!(
            avg_intra < avg_inter,
            "Intra-topic avg Hamming ({:.1}) should be less than inter-topic ({:.1})",
            avg_intra,
            avg_inter,
        );
        // Inter-topic with disjoint dim ranges should be near-random (~32 bits)
        assert!(
            avg_inter > 28.0,
            "Inter-topic avg Hamming should be > 28 (near-random), got {:.1}",
            avg_inter,
        );
    }

    /// Weight magnitude affects SimHash — heavy dims dominate.
    #[test]
    fn test_simhash_weight_sensitivity() {
        // Two vectors share dims 0..50 but differ on dim 100
        let mut v1: Vec<(u32, f32)> = (0..50).map(|d| (d, 0.1)).collect();
        v1.push((100, 5.0)); // heavy dim A

        let mut v2: Vec<(u32, f32)> = (0..50).map(|d| (d, 0.1)).collect();
        v2.push((200, 5.0)); // heavy dim B

        let mut v3: Vec<(u32, f32)> = (0..50).map(|d| (d, 0.1)).collect();
        v3.push((100, 5.0)); // heavy dim A (same as v1)

        let h1 = simhash_from_sparse_vector(&v1, 0.0, 5.0);
        let h2 = simhash_from_sparse_vector(&v2, 0.0, 5.0);
        let h3 = simhash_from_sparse_vector(&v3, 0.0, 5.0);

        assert_eq!(h1, h3, "Identical vectors should produce identical hashes");
        assert!(
            hamming_distance(h1, h2) > 0,
            "Different heavy dims should produce different hashes"
        );
    }

    /// Empty vector produces zero hash.
    #[test]
    fn test_simhash_empty() {
        assert_eq!(simhash_from_sparse_vector(&[], 0.0, 5.0), 0);
    }

    /// Majority SimHash: basic correctness.
    #[test]
    fn test_majority_simhash_basic() {
        // All same hash → majority = that hash
        let hashes = vec![0xFF00FF00_FF00FF00u64; 5];
        assert_eq!(majority_simhash(&hashes), 0xFF00FF00_FF00FF00);

        // 3 votes for bit 0, 2 votes against → bit 0 set
        let hashes = vec![0b1, 0b1, 0b1, 0b0, 0b0];
        assert_eq!(majority_simhash(&hashes) & 1, 1);

        // 2 votes for bit 0, 3 votes against → bit 0 unset
        let hashes = vec![0b1, 0b1, 0b0, 0b0, 0b0];
        assert_eq!(majority_simhash(&hashes) & 1, 0);
    }

    /// Stafford mix produces distinct values for adjacent dim IDs.
    #[test]
    fn test_stafford_mix_avalanche() {
        let h0 = stafford_mix(0);
        let h1 = stafford_mix(1);
        let h2 = stafford_mix(2);

        // Adjacent inputs should produce very different outputs (good avalanche)
        assert!(hamming_distance(h0, h1) > 20, "Poor avalanche: 0 vs 1");
        assert!(hamming_distance(h1, h2) > 20, "Poor avalanche: 1 vs 2");

        // All distinct
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    /// stafford_mix(0) must not be zero (fixed-point bug) and bit 0 must be active.
    #[test]
    fn test_stafford_mix_no_fixed_point() {
        assert_ne!(
            stafford_mix(0),
            0,
            "stafford_mix(0) must not be a fixed point"
        );

        // Bit 0 should be set for at least some dim_ids (not always zero)
        let bit0_set = (0..100).filter(|&d| stafford_mix(d) & 1 != 0).count();
        assert!(
            bit0_set > 30,
            "Bit 0 should be set ~50% of the time, got {}/100",
            bit0_set
        );
    }
}
