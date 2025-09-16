//! Fixed-width ranks of sorted subsets of a universe of at most 64 values.
//! Cardinality is supplied by the owner; no per-set header is needed.
const fn coefficients() -> [[u64; 65]; 65] {
    let mut table = [[0; 65]; 65];
    let mut n = 0;
    while n <= 64 {
        table[n][0] = 1;
        table[n][n] = 1;
        let mut k = 1;
        while k < n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
}
static CHOOSE: [[u64; 65]; 65] = coefficients();

pub(crate) fn width(universe: usize, count: usize) -> u8 {
    (64 - (CHOOSE[universe][count] - 1).leading_zeros()) as u8
}
#[cfg(test)]
pub(crate) fn valid(rank: u64, universe: usize, count: usize) -> bool {
    count <= universe && universe <= 64 && rank < CHOOSE[universe][count]
}
#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) fn rank(ids: &[u64]) -> u64 {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| CHOOSE[id as usize][i + 1])
        .sum()
}
/// Decode a writer-produced rank in ascending ID order.
pub(crate) fn decode(mut rank: u64, universe: usize, out: &mut [u32]) {
    let mut limit = universe;
    for k in (1..=out.len()).rev() {
        if k == 1 {
            out[0] = rank as u32;
            break;
        }
        let mut low = k - 1;
        let mut high = limit;
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            if CHOOSE[middle][k] <= rank {
                low = middle;
            } else {
                high = middle;
            }
        }
        out[k - 1] = low as u32;
        rank -= CHOOSE[low][k];
        limit = low;
    }
}
pub(crate) fn read(bytes: &[u8], bit: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let start = bit / 8;
    let len = (bit % 8 + usize::from(width)).div_ceil(8);
    let mut data = [0u8; 16];
    data[..len].copy_from_slice(&bytes[start..start + len]);
    ((u128::from_le_bytes(data) >> (bit % 8)) & ((1u128 << width) - 1)) as u64
}
#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) fn append(bytes: &mut Vec<u8>, bit: &mut usize, rank: u64, width: u8) {
    let end = *bit + usize::from(width);
    bytes.resize(end.div_ceil(8), 0);
    for shift in 0..usize::from(width) {
        bytes[(*bit + shift) / 8] |= (((rank >> shift) & 1) as u8) << ((*bit + shift) % 8);
    }
    *bit = end;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subset_ranks_round_trip_every_small_set_and_wide_extremes() {
        for universe in 0..=12 {
            for mask in 0u64..1 << universe {
                let ids: Vec<_> = (0..universe)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| i as u64)
                    .collect();
                let rank = rank(&ids);
                assert!(valid(rank, universe, ids.len()));
                let mut out = vec![0; ids.len()];
                decode(rank, universe, &mut out);
                assert_eq!(out.iter().map(|&i| u64::from(i)).collect::<Vec<_>>(), ids);
            }
        }
        let mut bytes = Vec::new();
        let mut bit = 0;
        for (k, &combinations) in CHOOSE[64].iter().enumerate() {
            let ids: Vec<_> = (64 - k..64).map(|i| i as u64).collect();
            let rank = rank(&ids);
            let width = width(64, k);
            let start = bit;
            append(&mut bytes, &mut bit, rank, width);
            assert_eq!(read(&bytes, start, width), rank);
            let mut out = vec![0; k];
            decode(rank, 64, &mut out);
            assert_eq!(out.iter().map(|&i| u64::from(i)).collect::<Vec<_>>(), ids);
            assert!(!valid(combinations, 64, k));
        }
    }
}
