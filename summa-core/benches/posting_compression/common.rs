//! Common utilities for posting compression benchmarks

#![allow(dead_code)]

use summa_core::structures::{
    EliasFanoPostingList, HorizontalBP128PostingList, OptP4DPostingList, PartitionedEFPostingList,
    RoaringPostingList, RoundedBP128PostingList, VerticalBP128PostingList,
};

/// Distribution types for doc_id generation
#[derive(Clone, Copy, Debug)]
pub enum Distribution {
    /// Random gaps with given density (count/universe ratio)
    Sparse, // 1% density - rare terms
    Medium,     // 10% density - typical terms
    Dense,      // 50% density - common terms
    Clustered,  // Grouped in clusters (locality)
    Sequential, // Consecutive doc_ids (best case for delta)
}

impl Distribution {
    pub fn name(&self) -> &'static str {
        match self {
            Distribution::Sparse => "sparse_1pct",
            Distribution::Medium => "medium_10pct",
            Distribution::Dense => "dense_50pct",
            Distribution::Clustered => "clustered",
            Distribution::Sequential => "sequential",
        }
    }

    pub fn density(&self) -> f64 {
        match self {
            Distribution::Sparse => 0.01,
            Distribution::Medium => 0.10,
            Distribution::Dense => 0.50,
            Distribution::Clustered => 0.10,
            Distribution::Sequential => 1.0,
        }
    }
}

/// Simple LCG for reproducible random numbers
fn next_rand_with_state(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Generate posting list with given distribution
pub fn generate_postings(count: usize, dist: Distribution) -> (Vec<u32>, Vec<u32>) {
    let mut doc_ids = Vec::with_capacity(count);
    let mut term_freqs = Vec::with_capacity(count);
    let mut state = 12345u64;

    let mut next_rand = || next_rand_with_state(&mut state);

    match dist {
        Distribution::Sparse => {
            // 1% density - large gaps
            let mut current_doc = 0u32;
            for _ in 0..count {
                let gap = ((next_rand() >> 33) % 200) as u32 + 1;
                current_doc += gap;
                doc_ids.push(current_doc);
            }
        }
        Distribution::Medium => {
            // 10% density - medium gaps
            let mut current_doc = 0u32;
            for _ in 0..count {
                let gap = ((next_rand() >> 33) % 20) as u32 + 1;
                current_doc += gap;
                doc_ids.push(current_doc);
            }
        }
        Distribution::Dense => {
            // 50% density - small gaps
            let mut current_doc = 0u32;
            for _ in 0..count {
                let gap = ((next_rand() >> 33) % 4) as u32 + 1;
                current_doc += gap;
                doc_ids.push(current_doc);
            }
        }
        Distribution::Clustered => {
            // Clustered - groups of consecutive docs with gaps between
            let mut current_doc = 0u32;
            let cluster_size = 50;
            let mut in_cluster = 0;
            for _ in 0..count {
                if in_cluster < cluster_size {
                    current_doc += 1;
                    in_cluster += 1;
                } else {
                    let cluster_gap = ((next_rand() >> 33) % 1000) as u32 + 100;
                    current_doc += cluster_gap;
                    in_cluster = 0;
                }
                doc_ids.push(current_doc);
            }
        }
        Distribution::Sequential => {
            // Consecutive doc_ids starting from random offset
            let start = (next_rand() >> 33) as u32 % 1_000_000;
            for i in 0..count {
                doc_ids.push(start + i as u32);
            }
        }
    }

    // TF follows Zipf-like distribution (mostly 1s)
    for _ in 0..count {
        let tf = if (next_rand() >> 40) % 10 < 7 {
            1
        } else {
            ((next_rand() >> 45) % 5 + 1) as u32
        };
        term_freqs.push(tf);
    }

    (doc_ids, term_freqs)
}

/// Compression stats for a single test case
pub struct CompressionResult {
    pub raw_bytes: usize,
    pub horiz_bp128_bytes: usize,
    pub horiz_bp128_rounded_bytes: usize,
    pub vert_bp128_bytes: usize,
    pub elias_fano_bytes: usize,
    pub partitioned_ef_bytes: usize,
    pub roaring_bytes: usize,
    pub opt_p4d_bytes: usize,
}

impl CompressionResult {
    pub fn horiz_bp128_ratio(&self) -> f64 {
        self.horiz_bp128_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    pub fn horiz_bp128_rounded_ratio(&self) -> f64 {
        self.horiz_bp128_rounded_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    pub fn vert_bp128_ratio(&self) -> f64 {
        self.vert_bp128_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    pub fn elias_fano_ratio(&self) -> f64 {
        self.elias_fano_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    pub fn partitioned_ef_ratio(&self) -> f64 {
        self.partitioned_ef_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    pub fn roaring_ratio(&self) -> f64 {
        self.roaring_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
    pub fn opt_p4d_ratio(&self) -> f64 {
        self.opt_p4d_bytes as f64 / self.raw_bytes as f64 * 100.0
    }
}

/// Measure compression for all formats
pub fn measure_compression(doc_ids: &[u32], term_freqs: &[u32]) -> CompressionResult {
    let raw_bytes = doc_ids.len() * 8; // 4 bytes doc_id + 4 bytes tf

    let horiz_bp128 = HorizontalBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let mut bp_buf = Vec::new();
    horiz_bp128.serialize(&mut bp_buf).unwrap();

    // Rounded bitpacking version (trades space for speed)
    let rounded_bp128 = RoundedBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let mut bp_rounded_buf = Vec::new();
    rounded_bp128.serialize(&mut bp_rounded_buf).unwrap();

    let vert_bp128 = VerticalBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let mut simd_buf = Vec::new();
    vert_bp128.serialize(&mut simd_buf).unwrap();

    let elias_fano = EliasFanoPostingList::from_postings(doc_ids, term_freqs);
    let mut ef_buf = Vec::new();
    elias_fano.serialize(&mut ef_buf).unwrap();

    let partitioned_ef = PartitionedEFPostingList::from_postings(doc_ids, term_freqs);
    let mut pef_buf = Vec::new();
    partitioned_ef.serialize(&mut pef_buf).unwrap();

    let roaring = RoaringPostingList::from_postings(doc_ids, term_freqs);
    let mut roar_buf = Vec::new();
    roaring.serialize(&mut roar_buf).unwrap();

    let opt_p4d = OptP4DPostingList::from_postings(doc_ids, term_freqs, 1.0);
    let mut opt_p4d_buf = Vec::new();
    opt_p4d.serialize(&mut opt_p4d_buf).unwrap();

    CompressionResult {
        raw_bytes,
        horiz_bp128_bytes: bp_buf.len(),
        horiz_bp128_rounded_bytes: bp_rounded_buf.len(),
        vert_bp128_bytes: simd_buf.len(),
        elias_fano_bytes: ef_buf.len(),
        partitioned_ef_bytes: pef_buf.len(),
        roaring_bytes: roar_buf.len(),
        opt_p4d_bytes: opt_p4d_buf.len(),
    }
}

/// Format rate for display
pub fn format_rate(rate: f64) -> String {
    if rate >= 1000.0 {
        format!("{:5.1}M", rate / 1000.0)
    } else if rate >= 1.0 {
        format!("{:5.1}K", rate)
    } else {
        format!("{:5.2}", rate * 1000.0)
    }
}

/// Find index of maximum value
pub fn find_best_idx(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find index of minimum value
pub fn find_min_idx(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Format names for all posting list types (8 formats total)
pub const FORMAT_NAMES: [&str; 8] = [
    "BP128",     // Horizontal bitpacking, 128-int blocks
    "BP128-Rnd", // Horizontal bitpacking with rounded bit widths (8/16/32)
    "VBP128",    // Vertical (SIMD) bitpacking, 128-int blocks
    "EF",        // Elias-Fano monotone sequence encoding
    "PEF",       // Partitioned Elias-Fano (better for skipping)
    "Roaring",   // Roaring bitmaps (hybrid array/bitmap/RLE)
    "OptP4D",    // Optimized Patched Frame-of-Reference Delta
    "Simple16",  // Simple-16 variable-length encoding (placeholder)
];

/// Short format names for table headers
pub const FORMAT_SHORT: [&str; 6] = ["BP128", "VBP128", "EF", "PEF", "Roaring", "OptP4D"];

/// Short format names including rounded bitpacking variant (7 formats)
pub const FORMAT_SHORT_7: [&str; 7] = [
    "BP128", "BP-Rnd", "VBP128", "EF", "PEF", "Roaring", "OptP4D",
];
