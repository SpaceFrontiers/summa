//! Standalone resident-row codec comparison; input is count-prefixed BMPA rows.
//! rustc --edition=2024 -O --cfg 'feature="native"' measure.rs -o measure
#![allow(dead_code)]
use std::{hint::black_box, time::Instant};
#[path = "../../../../summa-core/src/structures/postings/sparse/dimensions.rs"]
mod dimensions;
mod structures {
    pub(crate) mod postings {
        pub(crate) use crate::dimensions as sparse_dimensions;
    }
}
type Result<T> = std::result::Result<T, String>;
fn corrupt(message: &str) -> String {
    message.into()
}
#[path = "../../../../summa-core/src/segment/bmp_forward/codec.rs"]
mod codec;
fn raw(bytes: &[u8]) -> impl Iterator<Item = (u32, u8)> + '_ {
    bytes
        .chunks_exact(5)
        .map(|b| (u32::from_le_bytes(b[..4].try_into().unwrap()), b[4]))
}
fn score(values: impl Iterator<Item = (u32, u8)>, mut query: &[(u32, u16)]) -> u32 {
    let mut units = 0u32;
    for (dimension, impact) in values {
        while query.first().is_some_and(|&(dim, _)| dim < dimension) {
            query = &query[1..];
        }
        for &(_, weight) in query.iter().take_while(|&&(dim, _)| dim == dimension) {
            units = units
                .checked_add(u32::from(impact) * u32::from(weight))
                .unwrap();
        }
    }
    units
}
fn score_compact(vector: codec::ForwardVector<'_>, mut query: &[(u32, u16)]) -> u32 {
    vector
        .fold(Some(0u32), |units, dimension, impact| {
            let mut units = units?;
            while query.first().is_some_and(|&(dim, _)| dim < dimension) {
                query = &query[1..];
            }
            for &(_, weight) in query.iter().take_while(|&&(dim, _)| dim == dimension) {
                units = units.checked_add(u32::from(impact) * u32::from(weight))?;
            }
            Some(units)
        })
        .unwrap()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let input = std::fs::read(&args[1]).unwrap();
    let mut rest = input.as_slice();
    let mut before = Vec::new();
    let mut after = Vec::new();
    let mut tags = [0usize; 4];
    let mut writer = codec::RowWriter::default();
    let mut output = Vec::new();
    let mut query = Vec::new();
    for i in 0..16 {
        query.push((i * 6617, (i * 13 + 1) as u16));
    }
    let begin = Instant::now();
    while !rest.is_empty() {
        let count = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        let row = &rest[4..4 + count * 5];
        rest = &rest[4 + count * 5..];
        let mut encoded = Vec::new();
        for (dim, impact) in raw(row) {
            writer.push(dim, impact, &mut encoded).unwrap();
        }
        writer.finish(&mut encoded).unwrap();
        let vector = codec::ForwardVector(&encoded);
        assert_eq!(
            raw(row).collect::<Vec<_>>(),
            vector.iter().collect::<Vec<_>>()
        );
        assert_eq!(score(raw(row), &query), score_compact(vector, &query));
        let mut packet = &encoded[..encoded.len() - 4];
        while !packet.is_empty() {
            tags[packet[1] as usize] += 1;
            let extent = 4
                + u16::from_le_bytes(packet[2..4].try_into().unwrap()) as usize
                + packet[0] as usize;
            packet = &packet[extent..];
        }
        output.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        output.extend_from_slice(&encoded);
        before.push(row);
        after.push(encoded);
    }
    let prepare = begin.elapsed().as_secs_f64();
    std::fs::write(&args[2], output).unwrap();
    println!(
        "rows={} raw_bytes={} packet_bytes={} tags={:?} prepare_and_verify_s={:.6} row_writer_bytes={}",
        before.len(),
        before.iter().map(|r| r.len()).sum::<usize>(),
        after.iter().map(Vec::len).sum::<usize>(),
        tags,
        prepare,
        std::mem::size_of::<codec::RowWriter>()
    );
    let mode = args.get(3).map(String::as_str).unwrap_or("both");
    if mode == "encode" {
        return;
    }
    for pass in 0..7 {
        for compact in if pass % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            if mode == "raw" && compact || mode == "gap" && !compact {
                continue;
            }
            let begin = Instant::now();
            let mut checksum = 0u64;
            for _ in 0..50 {
                for i in 0..before.len() {
                    let units = if compact {
                        score_compact(
                            codec::ForwardVector(black_box(&after[i])),
                            black_box(&query),
                        )
                    } else {
                        score(raw(black_box(before[i])), black_box(&query))
                    };
                    checksum += u64::from(units);
                }
            }
            println!(
                "pass={} codec={} ns_per_row={:.3} checksum={}",
                pass,
                if compact { "gap" } else { "raw" },
                begin.elapsed().as_nanos() as f64 / (50 * before.len()) as f64,
                black_box(checksum)
            );
        }
    }
}
