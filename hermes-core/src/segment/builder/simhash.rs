/// Compute SimHash from a sparse vector's dimension IDs and weights.
///
/// Each (dim_id, weight) pair contributes to a 64-bit fingerprint:
/// - Hash dim_id with Stafford mix13 to get a 64-bit pseudo-random mask
/// - For each bit, accumulate +weight or -weight based on that bit
/// - Final hash: bit i = 1 iff accumulator[i] > 0
///
/// Documents with similar dimension sets produce similar hashes
/// (small Hamming distance), enabling block-reorder clustering.
#[inline]
pub fn simhash_from_sparse_vector(entries: &[(u32, f32)]) -> u64 {
    let mut acc = [0.0f64; 64];
    for &(dim, weight) in entries {
        // Stafford mix13 variant — good avalanche for 32→64 bit expansion
        let mut h = dim as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 32;
        h = h.wrapping_mul(0x6c62272e07bb0142);
        let w = weight as f64;
        // Process 4 bits at a time to reduce loop overhead
        let mut mask = h;
        for chunk in acc.chunks_exact_mut(4) {
            chunk[0] += if mask & 1 != 0 { w } else { -w };
            chunk[1] += if mask & 2 != 0 { w } else { -w };
            chunk[2] += if mask & 4 != 0 { w } else { -w };
            chunk[3] += if mask & 8 != 0 { w } else { -w };
            mask >>= 4;
        }
    }
    let mut hash = 0u64;
    for (bit, &a) in acc.iter().enumerate() {
        if a > 0.0 {
            hash |= 1u64 << bit;
        }
    }
    hash
}
