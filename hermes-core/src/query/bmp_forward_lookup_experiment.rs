//! Controlled query-table and validation experiments; no production dispatch.
use super::*;
use crate::directories::{Directory, DirectoryWriter, MmapDirectory};
use crate::segment::bmp_forward::ForwardVector;
use crate::segment::reorder::{BpGranularity, reorder_bmp_field};
use crate::segment::{OffsetWriter, build_bmp_blob};
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const DOCS: u32 = 8192;
const BLOCK_SIZE: u32 = 32;
const MAX_TABLE_DIMS: u32 = 105_879;

fn refresh_table(table: &mut [u32], previous: &[(u32, u16)], query: &[(u32, u16)]) {
    for &(dimension, _) in previous {
        table[dimension as usize] = 0;
    }
    for &(dimension, weight) in query {
        table[dimension as usize] += u32::from(weight);
    }
}

// Experimental table dot product, using the same quantized integer arithmetic.
// `validate` fuses selected-payload validation with the table read. The benchmark
// obtains the raw view through an already validated fixture without widening
// production APIs. Validation is deliberately repeated inside the timed kernel.
fn lookup_units(vector: ForwardVector<'_>, table: &[u32], validate: bool) -> crate::Result<u32> {
    let mut previous = None;
    let mut sum = 0u32;
    for (dimension, impact) in vector.iter() {
        let weight = *table.get(dimension as usize).ok_or_else(|| {
            crate::Error::Corruption("experimental forward dimension outside table".into())
        })?;
        if validate && (impact == 0 || previous.is_some_and(|p| p > dimension)) {
            return Err(crate::Error::Corruption(
                "experimental forward order or impact".into(),
            ));
        }
        previous = Some(dimension);
        sum = sum
            .checked_add(u32::from(impact) * weight)
            .ok_or_else(|| crate::Error::Query("experimental forward score overflow".into()))?;
    }
    Ok(sum)
}

async fn mapped_fixture(dir: &MmapDirectory, dims: u32) -> BmpIndex {
    let mut postings = rustc_hash::FxHashMap::default();
    for doc in 0..DOCS {
        // Adjacent logical documents belong to different term clusters. Record
        // BP makes physical neighbors coherent while forward values stay logical.
        let cluster = (doc * 997) % 256;
        for k in 0..64 {
            let dimension = (cluster * 257 + k * 61) % dims;
            postings.entry(dimension).or_insert_with(Vec::new).push((
                doc,
                0,
                0.2 + (k % 13) as f32 * 0.1,
            ));
        }
    }
    let mut writer = dir
        .streaming_writer_cold(Path::new("original"))
        .await
        .unwrap();
    build_bmp_blob(
        postings,
        BLOCK_SIZE,
        4,
        0.0,
        None,
        dims,
        5.0,
        0,
        true,
        &mut writer,
    )
    .unwrap();
    writer.flush().unwrap();
    writer.finish().unwrap();
    let handle = dir.open_read(Path::new("original")).await.unwrap();
    let len = handle.len();
    BmpIndex::parse(handle, 0, len, DOCS, DOCS).unwrap()
}

async fn record_bp_fixture(dir: &MmapDirectory, source: &BmpIndex) -> BmpIndex {
    let writer = OffsetWriter::new(dir.streaming_writer_cold(Path::new("bp")).await.unwrap());
    let (writer, _, converged) = reorder_bmp_field(
        &[(source.clone(), 0)],
        0,
        "forward-experiment",
        "sparse",
        source.dims(),
        BLOCK_SIZE as usize,
        4,
        5.0,
        DOCS,
        128 * 1024 * 1024,
        Default::default(),
        None,
        BpGranularity::Records,
        dir.root().join("bp-scratch"),
        true,
        writer,
        Vec::new(),
        None,
    )
    .unwrap();
    assert!(converged);
    writer.finish().unwrap();
    let handle = dir.open_read(Path::new("bp")).await.unwrap();
    let len = handle.len();
    let result = BmpIndex::parse(handle, 0, len, DOCS, DOCS).unwrap();
    // Forward ownership and byte-copy invariants are checked by the storage
    // tests; here ensure the physical locality experiment really changed order.
    let moved = (0..result.num_virtual_docs)
        .filter(|&slot| {
            slot >= source.num_virtual_docs
                || result.virtual_to_doc(slot) != source.virtual_to_doc(slot)
        })
        .count();
    assert!(moved > DOCS as usize / 2);
    eprintln!("lookup_fixture dims={} moved_slots={moved}", source.dims());
    result
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure_layout(index: &BmpIndex, layout: &str) {
    let terms: Vec<_> = (0..64)
        .map(|i| (i * 61, 0.1 + (i % 7) as f32 * 0.1))
        .collect();
    let prepared = prepare_bmp_query(index.dims(), &terms, &terms)
        .unwrap()
        .unwrap();
    let mask = !prepared.phase1_mask & valid_query_bits(prepared.query_by_dim_u16.len());
    let remaining: Vec<_> = prepared
        .query_by_dim_u16
        .iter()
        .enumerate()
        .filter_map(|(i, &term)| (mask & (1 << i) != 0).then_some(term))
        .collect();
    let forward = index.forward().unwrap();
    let validated = forward.validate_payload().unwrap();
    assert!(index.dims() <= MAX_TABLE_DIMS);
    let mut table = vec![0u32; index.dims() as usize];
    refresh_table(&mut table, &[], &remaining);
    let alternate: Vec<_> = remaining
        .iter()
        .map(|&(dim, w)| ((dim + 257) % index.dims(), w))
        .collect();
    let active: Vec<_> = (0..index.num_blocks)
        .filter_map(|id| {
            let block = index.parse_block(id).unwrap();
            let present =
                prepared
                    .query_by_dim_u16
                    .iter()
                    .enumerate()
                    .fold(0, |bits, (i, &(dim, _))| {
                        bits | if block.find_dimension(dim).is_some() {
                            1u64 << i
                        } else {
                            0
                        }
                    })
                    & mask;
            (present != 0).then_some((id, block, present))
        })
        .collect();
    assert!(!active.is_empty());
    let blocks: Vec<_> = active
        .iter()
        .step_by(active.len().div_ceil(64))
        .copied()
        .collect();
    let mean_terms = blocks
        .iter()
        .map(|(_, _, mask)| mask.count_ones() as f64)
        .sum::<f64>()
        / blocks.len() as f64;
    eprintln!(
        "lookup_active layout={layout} dims={} active_blocks={} sampled={} mean_terms={mean_terms:.2}",
        index.dims(),
        active.len(),
        blocks.len()
    );
    let mut setup = Vec::new();
    let mut fresh = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        for i in 0..100 {
            let (previous, query) = if i % 2 == 0 {
                (&remaining, &alternate)
            } else {
                (&alternate, &remaining)
            };
            refresh_table(black_box(&mut table), black_box(previous), black_box(query));
            black_box(&table);
        }
        setup.push(start.elapsed().as_secs_f64() * 1e9 / 100.0);
        let start = Instant::now();
        for _ in 0..100 {
            let mut table = vec![0u32; black_box(index.dims() as usize)];
            refresh_table(&mut table, &[], black_box(&remaining));
            black_box(table);
        }
        fresh.push(start.elapsed().as_secs_f64() * 1e9 / 100.0);
    }
    eprintln!(
        "lookup_setup layout={layout} dims={} bytes={} refresh_ns={:.1} fresh_ns={:.1}",
        index.dims(),
        table.len() * 4,
        median(&mut setup),
        median(&mut fresh)
    );
    for survivors in [1usize, 2, 4, 8, 32] {
        let slots: Vec<_> = (0..survivors)
            .map(|i| (i * 19) % BLOCK_SIZE as usize)
            .collect();
        for &(block_id, block, present) in &blocks {
            let mut expected = [0u32; 256];
            let faults = score_block_bsearch_int(
                block,
                &prepared.query_by_dim_u16,
                present,
                false,
                &mut expected,
                &mut [0; 4],
                BLOCK_SIZE as usize,
            );
            assert_eq!(faults.corrupt_terms + faults.dropped_postings, 0);
            for &slot in &slots {
                let (doc, ordinal) = index.virtual_to_doc(block_id * BLOCK_SIZE + slot as u32);
                let id = forward
                    .find(crate::segment::logical_address::LogicalUnit { doc, ordinal })
                    .unwrap();
                assert_eq!(
                    score_forward_units(forward.vector(id).unwrap(), &remaining).unwrap(),
                    expected[slot]
                );
                assert_eq!(
                    lookup_units(forward.vector(id).unwrap(), &table, false).unwrap(),
                    expected[slot]
                );
                assert_eq!(
                    lookup_units(validated.vector(id), &table, true).unwrap(),
                    expected[slot]
                );
            }
        }
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for sample in 0..11 {
            for offset in 0..4 {
                let kernel = (sample + offset) % 4;
                let start = Instant::now();
                for _ in 0..10 {
                    for &(block_id, block, present) in &blocks {
                        if kernel == 0 {
                            let mut acc = [0; 256];
                            black_box(score_block_bsearch_int(
                                block,
                                black_box(&prepared.query_by_dim_u16),
                                present,
                                false,
                                &mut acc,
                                &mut [0; 4],
                                BLOCK_SIZE as usize,
                            ));
                            black_box(acc);
                            continue;
                        }
                        for &slot in black_box(&slots) {
                            let (doc, ordinal) =
                                index.virtual_to_doc(block_id * BLOCK_SIZE + slot as u32);
                            let id = forward
                                .find(crate::segment::logical_address::LogicalUnit { doc, ordinal })
                                .unwrap();
                            black_box(match kernel {
                                1 => score_forward_units(
                                    forward.vector(id).unwrap(),
                                    black_box(&remaining),
                                )
                                .unwrap(),
                                2 => lookup_units(
                                    forward.vector(id).unwrap(),
                                    black_box(&table),
                                    false,
                                )
                                .unwrap(),
                                _ => lookup_units(validated.vector(id), black_box(&table), true)
                                    .unwrap(),
                            });
                        }
                    }
                }
                times[kernel]
                    .push(start.elapsed().as_secs_f64() * 1e9 / (blocks.len() * 10) as f64);
            }
        }
        eprintln!(
            "lookup_experiment layout={layout} dims={} survivors={survivors} inverted_ns={:.1} cursor_ns={:.1} lookup_ns={:.1} fused_lookup_ns={:.1}",
            index.dims(),
            median(&mut times[0]),
            median(&mut times[1]),
            median(&mut times[2]),
            median(&mut times[3])
        );
    }
}
