//! Production-shaped comparison of BMP document payloads:
//!
//! - current Flat-Inv (`u32` posting starts, sparse `(slot, impact)` pairs);
//! - compact sparse Flat-Inv (adaptive `u16`/`u32` byte offsets and u4 maxima);
//! - adaptive Flat-Inv (compact directory plus dense 32-impact rows whenever
//!   they are no larger than sparse pairs);
//! - document-major forward payloads retained as a historical comparison.
//!
//! Run:
//! `cargo bench -p summa-core --bench bmp_payload_layout`
//!
//! Scale the corpus with `BMP_LAYOUT_BLOCKS` and `BMP_LAYOUT_TERMS_PER_DOC`.
//! By default both diffuse (2,048 topic dimensions) and BP-clustered (128)
//! workloads run. Set `BMP_LAYOUT_TOPIC_VOCAB` to benchmark one custom
//! locality level.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[allow(dead_code)]
#[path = "../src/segment/bmp_adaptive.rs"]
mod bmp_adaptive;
use bmp_adaptive::{AdaptiveBlock, AdaptiveEncodeScratch, AdaptivePostings};

const BLOCK_SIZE: usize = 32;
const VOCABULARY: u32 = 105_879;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}

#[derive(Clone, Copy)]
struct Posting {
    slot: u8,
    impact: u8,
}

struct FlatBlock {
    dimensions: Vec<u32>,
    posting_starts: Vec<u32>,
    maxima: Vec<u8>,
    postings: Vec<Posting>,
}

struct AdaptiveFlatBlock {
    bytes: Vec<u8>,
    dense_terms: usize,
}

impl AdaptiveFlatBlock {
    fn from_flat(block: &FlatBlock, enable_dense_rows: bool) -> Self {
        let posting_counts: Vec<u32> = block
            .posting_starts
            .windows(2)
            .map(|range| range[1] - range[0])
            .collect();
        let mut sparse_postings = Vec::with_capacity(block.postings.len() * 2);
        for posting in &block.postings {
            sparse_postings.extend_from_slice(&[posting.slot, posting.impact]);
        }
        let mut bytes = Vec::new();
        let mut scratch = AdaptiveEncodeScratch::default();
        scratch
            .encode(
                BLOCK_SIZE,
                enable_dense_rows,
                &block.dimensions,
                &posting_counts,
                &block.maxima,
                &sparse_postings,
                &mut bytes,
            )
            .unwrap();
        let parsed = AdaptiveBlock::parse(&bytes, BLOCK_SIZE).unwrap();
        let dense_terms = parsed
            .terms()
            .filter(|&(_, _, postings)| matches!(postings, AdaptivePostings::Dense(_)))
            .count();
        Self { bytes, dense_terms }
    }

    fn wire_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn uses_narrow_offsets(&self) -> bool {
        u32::from_le_bytes(self.bytes[..4].try_into().unwrap()) & (1 << 31) == 0
    }
}

struct ForwardBlock {
    document_starts: Vec<u32>,
    dimensions: Vec<u32>,
    impacts: Vec<u8>,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn generate_blocks(
    count: usize,
    terms_per_document: usize,
    topic_vocabulary: u32,
) -> (Vec<FlatBlock>, Vec<ForwardBlock>) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut flat = Vec::with_capacity(count);
    let mut forward = Vec::with_capacity(count);
    for _ in 0..count {
        // These are blocks that D has already selected for one topical query.
        // Giving every measured block the same topic avoids timing easy
        // Flat-Inv misses that production traversal would prune before the
        // payload scorer.
        let topic_base = 0u32;
        let mut documents = Vec::with_capacity(BLOCK_SIZE);
        for _ in 0..BLOCK_SIZE {
            let mut document = Vec::with_capacity(terms_per_document);
            for term in 0..terms_per_document {
                let dimension = if term * 4 < terms_per_document * 3 {
                    (topic_base + rng.next() % topic_vocabulary) % VOCABULARY
                } else {
                    rng.next() % VOCABULARY
                };
                let impact = 1 + (rng.next() % 255) as u8;
                document.push((dimension, impact));
            }
            document.sort_unstable_by_key(|&(dimension, _)| dimension);
            document.dedup_by(|left, right| {
                if left.0 == right.0 {
                    right.1 = right.1.max(left.1);
                    true
                } else {
                    false
                }
            });
            documents.push(document);
        }

        let mut document_starts = Vec::with_capacity(BLOCK_SIZE + 1);
        let mut forward_dimensions = Vec::new();
        let mut forward_impacts = Vec::new();
        for document in &documents {
            document_starts.push(forward_dimensions.len() as u32);
            for &(dimension, impact) in document {
                forward_dimensions.push(dimension);
                forward_impacts.push(impact);
            }
        }
        document_starts.push(forward_dimensions.len() as u32);
        forward.push(ForwardBlock {
            document_starts,
            dimensions: forward_dimensions,
            impacts: forward_impacts,
        });

        let mut entries = Vec::new();
        for (slot, document) in documents.iter().enumerate() {
            entries.extend(
                document
                    .iter()
                    .map(|&(dimension, impact)| (dimension, slot as u8, impact)),
            );
        }
        entries.sort_unstable_by_key(|&(dimension, slot, _)| (dimension, slot));
        let mut dimensions = Vec::new();
        let mut posting_starts = Vec::new();
        let mut maxima = Vec::new();
        let mut postings = Vec::with_capacity(entries.len());
        let mut cursor = 0;
        while cursor < entries.len() {
            let dimension = entries[cursor].0;
            dimensions.push(dimension);
            posting_starts.push(postings.len() as u32);
            let mut maximum = 0;
            while cursor < entries.len() && entries[cursor].0 == dimension {
                maximum = maximum.max(entries[cursor].2);
                postings.push(Posting {
                    slot: entries[cursor].1,
                    impact: entries[cursor].2,
                });
                cursor += 1;
            }
            maxima.push(maximum);
        }
        posting_starts.push(postings.len() as u32);
        flat.push(FlatBlock {
            dimensions,
            posting_starts,
            maxima,
            postings,
        });
    }
    (flat, forward)
}

fn query(dimensions: usize, topic_vocabulary: u32) -> Vec<(u32, u16)> {
    let mut rng = Rng(0xd1b5_4a32_d192_ed03 ^ dimensions as u64);
    let mut query = Vec::with_capacity(dimensions);
    for index in 0..dimensions {
        let dimension = if index * 4 < dimensions * 3 {
            rng.next() % topic_vocabulary
        } else {
            rng.next() % VOCABULARY
        };
        query.push((dimension, 1 + (rng.next() % 16_383) as u16));
    }
    query.sort_unstable_by_key(|&(dimension, _)| dimension);
    query.dedup_by_key(|entry| entry.0);
    query
}

#[inline]
fn score_flat(block: &FlatBlock, query: &[(u32, u16)], scores: &mut [u32; BLOCK_SIZE]) {
    scores.fill(0);
    for &(dimension, weight) in query {
        let Ok(term) = block.dimensions.binary_search(&dimension) else {
            continue;
        };
        let start = block.posting_starts[term] as usize;
        let end = block.posting_starts[term + 1] as usize;
        for posting in &block.postings[start..end] {
            scores[posting.slot as usize] += u32::from(posting.impact) * u32::from(weight);
        }
    }
}

#[inline]
fn score_adaptive_flat(
    block: &AdaptiveFlatBlock,
    query: &[(u32, u16)],
    scores: &mut [u32; BLOCK_SIZE],
) {
    scores.fill(0);
    let decoded = AdaptiveBlock::parse(&block.bytes, BLOCK_SIZE).unwrap();
    for &(dimension, weight) in query {
        let term = if query.len() <= 8 {
            decoded.find_dimension_branching(dimension)
        } else {
            decoded.find_dimension(dimension)
        };
        let Some(term) = term else {
            continue;
        };
        match decoded.postings(term).unwrap() {
            AdaptivePostings::Dense(impacts) => {
                for (score, &impact) in scores.iter_mut().zip(impacts) {
                    *score += u32::from(impact) * u32::from(weight);
                }
            }
            AdaptivePostings::Sparse(postings) => {
                for posting in postings {
                    scores[posting.local_slot as usize] +=
                        u32::from(posting.impact) * u32::from(weight);
                }
            }
        }
    }
}

#[inline]
fn score_forward(block: &ForwardBlock, query: &[(u32, u16)], scores: &mut [u32; BLOCK_SIZE]) {
    scores.fill(0);
    for (document, score) in scores.iter_mut().enumerate() {
        let start = block.document_starts[document] as usize;
        let end = block.document_starts[document + 1] as usize;
        let dimensions = &block.dimensions[start..end];
        let impacts = &block.impacts[start..end];
        let (mut left, mut right) = (0usize, 0usize);
        while left < dimensions.len() && right < query.len() {
            match dimensions[left].cmp(&query[right].0) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    *score += u32::from(impacts[left]) * u32::from(query[right].1);
                    left += 1;
                    right += 1;
                }
            }
        }
    }
}

#[inline]
fn score_forward_lookup(
    block: &ForwardBlock,
    query_lookup: &[u16],
    scores: &mut [u32; BLOCK_SIZE],
) {
    scores.fill(0);
    for (document, score) in scores.iter_mut().enumerate() {
        let start = block.document_starts[document] as usize;
        let end = block.document_starts[document + 1] as usize;
        for (&dimension, &impact) in block.dimensions[start..end]
            .iter()
            .zip(&block.impacts[start..end])
        {
            let weight = query_lookup[dimension as usize];
            *score += u32::from(impact) * u32::from(weight);
        }
    }
}

fn flat_wire_bytes(blocks: &[FlatBlock]) -> usize {
    blocks
        .iter()
        .map(|block| {
            4 // num_terms
                + block.dimensions.len() * 4
                + block.posting_starts.len() * 4
                + block.maxima.len()
                + block.postings.len() * 2
        })
        .sum()
}

fn forward_wire_bytes(blocks: &[ForwardBlock]) -> usize {
    blocks
        .iter()
        .map(|block| {
            block.document_starts.len() * 4 + block.dimensions.len() * 4 + block.impacts.len()
        })
        .sum()
}

fn adaptive_wire_bytes(blocks: &[AdaptiveFlatBlock]) -> usize {
    blocks.iter().map(AdaptiveFlatBlock::wire_bytes).sum()
}

fn benchmark(c: &mut Criterion) {
    let block_count = env_usize("BMP_LAYOUT_BLOCKS", 2_048);
    let terms_per_document = env_usize("BMP_LAYOUT_TERMS_PER_DOC", 96);
    let requested_topic_vocabulary = std::env::var("BMP_LAYOUT_TOPIC_VOCAB")
        .ok()
        .and_then(|value| value.parse::<u32>().ok());
    let scenarios: Vec<(&str, u32)> = requested_topic_vocabulary.map_or_else(
        || vec![("diffuse", 2_048), ("clustered", 128)],
        |vocabulary| vec![("custom", vocabulary.clamp(1, VOCABULARY))],
    );

    for (scenario, topic_vocabulary) in scenarios {
        let (flat, forward) = generate_blocks(block_count, terms_per_document, topic_vocabulary);
        let compact: Vec<_> = flat
            .iter()
            .map(|block| AdaptiveFlatBlock::from_flat(block, false))
            .collect();
        let adaptive: Vec<_> = flat
            .iter()
            .map(|block| AdaptiveFlatBlock::from_flat(block, true))
            .collect();
        let flat_bytes = flat_wire_bytes(&flat);
        let compact_bytes = adaptive_wire_bytes(&compact);
        let adaptive_bytes = adaptive_wire_bytes(&adaptive);
        let forward_bytes = forward_wire_bytes(&forward);
        let total_terms: usize = flat.iter().map(|block| block.dimensions.len()).sum();
        let dense_terms: usize = adaptive.iter().map(|block| block.dense_terms).sum();
        let narrow_blocks = adaptive
            .iter()
            .filter(|block| block.uses_narrow_offsets())
            .count();
        eprintln!(
            "BMP payload layout [{scenario}]: blocks={block_count}, block_size={BLOCK_SIZE}, \
             topic_vocab={topic_vocabulary}, terms/doc≈{terms_per_document}, \
             Flat-Inv={flat_bytes} B, Compact-Sparse={compact_bytes} B ({:.3}x), \
             Adaptive-Flat={adaptive_bytes} B ({:.3}x), \
             Fwd={forward_bytes} B ({:.3}x), dense_terms={dense_terms}/{total_terms} ({:.2}%), \
             narrow_blocks={narrow_blocks}/{block_count}",
            compact_bytes as f64 / flat_bytes as f64,
            adaptive_bytes as f64 / flat_bytes as f64,
            forward_bytes as f64 / flat_bytes as f64,
            dense_terms as f64 * 100.0 / total_terms as f64,
        );

        let mut group = c.benchmark_group(format!("bmp_payload_layout/{scenario}"));
        group.throughput(Throughput::Elements(block_count as u64));
        for query_dimensions in [8, 32, 64] {
            let query = query(query_dimensions, topic_vocabulary);
            let mut query_lookup = vec![0u16; VOCABULARY as usize];
            for &(dimension, weight) in &query {
                query_lookup[dimension as usize] = weight;
            }

            let mut expected = [0u32; BLOCK_SIZE];
            let mut actual = [0u32; BLOCK_SIZE];
            for ((flat_block, compact_block), adaptive_block) in
                flat.iter().zip(&compact).zip(&adaptive)
            {
                score_flat(flat_block, &query, &mut expected);
                score_adaptive_flat(compact_block, &query, &mut actual);
                assert_eq!(
                    actual, expected,
                    "Compact sparse Flat-Inv changed scores in {scenario}/{query_dimensions}",
                );
                score_adaptive_flat(adaptive_block, &query, &mut actual);
                assert_eq!(
                    actual, expected,
                    "Adaptive Flat-Inv changed scores in {scenario}/{query_dimensions}",
                );
            }

            group.bench_with_input(
                BenchmarkId::new("flat_inv", query_dimensions),
                &query,
                |bencher, query| {
                    let mut scores = [0u32; BLOCK_SIZE];
                    bencher.iter(|| {
                        let mut checksum = 0u32;
                        for block in &flat {
                            score_flat(block, black_box(query), &mut scores);
                            checksum ^= scores.iter().copied().max().unwrap_or(0);
                        }
                        black_box(checksum)
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("compact_sparse", query_dimensions),
                &query,
                |bencher, query| {
                    let mut scores = [0u32; BLOCK_SIZE];
                    bencher.iter(|| {
                        let mut checksum = 0u32;
                        for block in &compact {
                            score_adaptive_flat(block, black_box(query), &mut scores);
                            checksum ^= scores.iter().copied().max().unwrap_or(0);
                        }
                        black_box(checksum)
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("adaptive_flat", query_dimensions),
                &query,
                |bencher, query| {
                    let mut scores = [0u32; BLOCK_SIZE];
                    bencher.iter(|| {
                        let mut checksum = 0u32;
                        for block in &adaptive {
                            score_adaptive_flat(block, black_box(query), &mut scores);
                            checksum ^= scores.iter().copied().max().unwrap_or(0);
                        }
                        black_box(checksum)
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("forward_merge", query_dimensions),
                &query,
                |bencher, query| {
                    let mut scores = [0u32; BLOCK_SIZE];
                    bencher.iter(|| {
                        let mut checksum = 0u32;
                        for block in &forward {
                            score_forward(block, black_box(query), &mut scores);
                            checksum ^= scores.iter().copied().max().unwrap_or(0);
                        }
                        black_box(checksum)
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("forward_lookup", query_dimensions),
                &query_lookup,
                |bencher, query_lookup| {
                    let mut scores = [0u32; BLOCK_SIZE];
                    bencher.iter(|| {
                        let mut checksum = 0u32;
                        for block in &forward {
                            score_forward_lookup(block, black_box(query_lookup), &mut scores);
                            checksum ^= scores.iter().copied().max().unwrap_or(0);
                        }
                        black_box(checksum)
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
