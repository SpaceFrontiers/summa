//! Pure block decompression benchmarks
//!
//! Measures actual SIMD bitpacking + delta decoding speed.
//! Expected: ~4-6 billion ints/sec (4-6K ints/μs) on modern CPUs with SSE/NEON.

use comfy_table::{Cell, Color, Table, presets::UTF8_FULL};
use std::hint::black_box;

use criterion::{Criterion, Throughput};
use std::time::Instant;
use summa_core::structures::simd::RoundedBitWidth;
use summa_core::structures::{HORIZONTAL_BP128_BLOCK_SIZE, pack_block, simd, unpack_block};

const BLOCK_SIZE: usize = HORIZONTAL_BP128_BLOCK_SIZE; // 128

/// SIMD-accelerated delta decode wrapper
#[inline]
fn delta_decode_simd(output: &mut [u32], deltas: &[u32], first_doc_id: u32, count: usize) {
    simd::delta_decode(output, deltas, first_doc_id, count);
}

/// Benchmark raw block unpacking (bitpacking only, no delta)
pub fn bench_block_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_unpack");

    // Test different bit widths
    for bit_width in [4u8, 8, 12, 16, 20, 24, 32] {
        // Create test data
        let max_val = if bit_width >= 32 {
            u32::MAX
        } else {
            (1u32 << bit_width) - 1
        };
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = if max_val == u32::MAX {
                (i as u32).wrapping_mul(7)
            } else {
                (i as u32 * 7) % (max_val + 1)
            };
        }

        // Pack the block
        let mut packed = Vec::new();
        pack_block(&values, bit_width, &mut packed);

        let num_blocks = 10000;
        group.throughput(Throughput::Elements((BLOCK_SIZE * num_blocks) as u64));

        group.bench_function(format!("{}bit", bit_width), |b| {
            b.iter(|| {
                let mut output = [0u32; BLOCK_SIZE];
                for _ in 0..num_blocks {
                    unpack_block(black_box(&packed), bit_width, &mut output);
                    black_box(&output);
                }
            })
        });
    }

    group.finish();
}

/// Benchmark block unpack + delta decode (full decompression pipeline)
pub fn bench_block_decompress_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_decompress_full");

    for bit_width in [4u8, 8, 12, 16] {
        // Create delta-encoded test data (typical posting list pattern)
        let max_delta = if bit_width >= 32 {
            u32::MAX
        } else {
            (1u32 << bit_width) - 1
        };
        let mut deltas = [0u32; BLOCK_SIZE];
        for (i, d) in deltas.iter_mut().enumerate() {
            *d = ((i as u32 * 3) % max_delta).max(1); // gaps >= 1
        }

        // Pack the deltas
        let mut packed = Vec::new();
        pack_block(&deltas, bit_width, &mut packed);

        let num_blocks = 10000;
        group.throughput(Throughput::Elements((BLOCK_SIZE * num_blocks) as u64));

        group.bench_function(format!("{}bit_unpack_delta", bit_width), |b| {
            b.iter(|| {
                let mut unpacked = [0u32; BLOCK_SIZE];
                let mut doc_ids = [0u32; BLOCK_SIZE];

                for block_idx in 0..num_blocks {
                    // Step 1: Unpack bits
                    unpack_block(black_box(&packed), bit_width, &mut unpacked);

                    // Step 2: Delta decode (prefix sum)
                    let first_doc = block_idx as u32 * 1000;
                    doc_ids[0] = first_doc;
                    for i in 1..BLOCK_SIZE {
                        doc_ids[i] = doc_ids[i - 1] + unpacked[i - 1] + 1;
                    }

                    black_box(&doc_ids);
                }
            })
        });
    }

    group.finish();
}

/// Measure and print raw decompression throughput
pub fn bench_raw_throughput(_c: &mut Criterion) {
    let iterations = 100_000;
    let total_ints = BLOCK_SIZE * iterations;

    // Collect results
    let mut results: Vec<(u8, f64, f64, f64, bool)> = Vec::new();

    for bit_width in [4u8, 8, 12, 16, 20, 24, 32] {
        // Create and pack test data
        let max_val = if bit_width >= 32 {
            u32::MAX
        } else {
            (1u32 << bit_width) - 1
        };
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = if max_val == u32::MAX {
                (i as u32).wrapping_mul(7)
            } else {
                (i as u32 * 7) % (max_val + 1)
            };
        }

        let mut packed = Vec::new();
        pack_block(&values, bit_width, &mut packed);

        // Measure unpack only
        let mut output = [0u32; BLOCK_SIZE];
        let start = Instant::now();
        for _ in 0..iterations {
            unpack_block(black_box(&packed), bit_width, &mut output);
            black_box(&output);
        }
        let unpack_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        // Measure unpack + scalar delta decode
        let mut doc_ids = [0u32; BLOCK_SIZE];
        let start = Instant::now();
        for block_idx in 0..iterations {
            unpack_block(black_box(&packed), bit_width, &mut output);
            let first_doc = block_idx as u32 * 1000;
            doc_ids[0] = first_doc;
            for i in 1..BLOCK_SIZE {
                doc_ids[i] = doc_ids[i - 1] + output[i - 1] + 1;
            }
            black_box(&doc_ids);
        }
        let scalar_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        // Measure unpack + SIMD delta decode
        let start = Instant::now();
        for block_idx in 0..iterations {
            unpack_block(black_box(&packed), bit_width, &mut output);
            let first_doc = block_idx as u32 * 1000;
            delta_decode_simd(&mut doc_ids, &output, first_doc, BLOCK_SIZE);
            black_box(&doc_ids);
        }
        let simd_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        let is_simd_optimized = bit_width == 8 || bit_width == 16 || bit_width == 32;
        results.push((
            bit_width,
            unpack_rate,
            scalar_rate,
            simd_rate,
            is_simd_optimized,
        ));
    }

    // Build table with comfy-table
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("Bits"),
        Cell::new("Unpack Only\n(bitpacking)"),
        Cell::new("Unpack+Scalar\n(+prefix sum)"),
        Cell::new("Unpack+SIMD\n(+SIMD prefix)"),
    ]);

    for (bit_width, unpack, scalar, simd, is_simd) in &results {
        let bits_label = if *is_simd {
            format!("{}*", bit_width)
        } else {
            format!("{}", bit_width)
        };

        let simd_cell = if *is_simd {
            Cell::new(format!("{:.2}", simd)).fg(Color::Green)
        } else {
            Cell::new(format!("{:.2}", simd))
        };

        table.add_row(vec![
            Cell::new(bits_label),
            Cell::new(format!("{:.2}", unpack)),
            Cell::new(format!("{:.2}", scalar)),
            simd_cell,
        ]);
    }

    println!("\n📊 RAW BLOCK DECOMPRESSION THROUGHPUT (Gints/s)");
    println!(
        "Block size: {} integers | Iterations: {}K blocks\n",
        BLOCK_SIZE,
        iterations / 1000
    );
    println!("{table}");
}

/// Benchmark rounded bitpacking vs exact bitpacking
///
/// Rounded bitpacking rounds bit widths to 8/16/32 for faster SIMD decoding.
/// Trades ~10-20% more space for significantly faster decode.
pub fn bench_rounded_bitpacking(_c: &mut Criterion) {
    let iterations = 100_000;
    let total_ints = BLOCK_SIZE * iterations;

    // Collect results: (exact_bits, rounded_bits, exact_rate, rounded_rate, space_overhead)
    let mut results: Vec<(u8, u8, f64, f64, f64)> = Vec::new();

    for exact_bits in [4u8, 6, 8, 10, 12, 14, 16, 20, 24] {
        let rounded = simd::round_bit_width(exact_bits);
        let rounded_width = RoundedBitWidth::from_exact(exact_bits);

        // Create test data that fits in exact_bits
        let max_val = if exact_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << exact_bits) - 1
        };
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as u32 * 7) % (max_val + 1);
        }

        // Pack with exact bitpacking
        let mut packed_exact = Vec::new();
        pack_block(&values, exact_bits, &mut packed_exact);

        // Pack with rounded bitpacking
        let mut packed_rounded = vec![0u8; BLOCK_SIZE * rounded_width.bytes_per_value()];
        simd::pack_rounded(&values[..], rounded_width, &mut packed_rounded);

        // Space overhead
        let exact_size = packed_exact.len();
        let rounded_size = packed_rounded.len();
        let space_overhead = if exact_size > 0 {
            (rounded_size as f64 / exact_size as f64 - 1.0) * 100.0
        } else {
            0.0
        };

        // Benchmark exact bitpacking decode
        let mut output = [0u32; BLOCK_SIZE];
        let start = Instant::now();
        for _ in 0..iterations {
            unpack_block(black_box(&packed_exact), exact_bits, &mut output);
            black_box(&output);
        }
        let exact_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        // Benchmark rounded bitpacking decode
        let start = Instant::now();
        for _ in 0..iterations {
            simd::unpack_rounded(
                black_box(&packed_rounded),
                rounded_width,
                &mut output,
                BLOCK_SIZE,
            );
            black_box(&output);
        }
        let rounded_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        results.push((
            exact_bits,
            rounded,
            exact_rate,
            rounded_rate,
            space_overhead,
        ));
    }

    // Build table
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("Exact\nBits"),
        Cell::new("Rounded\nBits"),
        Cell::new("Exact\n(Gints/s)"),
        Cell::new("Rounded\n(Gints/s)"),
        Cell::new("Speedup"),
        Cell::new("Space\nOverhead"),
    ]);

    for (exact, rounded, exact_rate, rounded_rate, overhead) in &results {
        let speedup = rounded_rate / exact_rate;
        let speedup_cell = if speedup > 1.5 {
            Cell::new(format!("{:.2}x", speedup)).fg(Color::Green)
        } else if speedup > 1.1 {
            Cell::new(format!("{:.2}x", speedup)).fg(Color::Yellow)
        } else {
            Cell::new(format!("{:.2}x", speedup))
        };

        table.add_row(vec![
            Cell::new(format!("{}", exact)),
            Cell::new(format!("{}", rounded)),
            Cell::new(format!("{:.2}", exact_rate)),
            Cell::new(format!("{:.2}", rounded_rate)),
            speedup_cell,
            Cell::new(format!("+{:.0}%", overhead)),
        ]);
    }

    println!("\n📊 ROUNDED vs EXACT BITPACKING THROUGHPUT");
    println!(
        "Block size: {} integers | Iterations: {}K blocks\n",
        BLOCK_SIZE,
        iterations / 1000
    );
    println!("{table}");
}

/// Benchmark fused rounded unpack + delta decode
pub fn bench_rounded_fused_delta(_c: &mut Criterion) {
    let iterations = 100_000;
    let total_ints = BLOCK_SIZE * iterations;

    // Collect results
    let mut results: Vec<(u8, f64, f64, f64)> = Vec::new();

    for rounded_bits in [8u8, 16] {
        let rounded_width = RoundedBitWidth::from_exact(rounded_bits);

        // Create delta-encoded test data
        let max_delta = (1u32 << rounded_bits) - 1;
        let mut deltas = [0u32; BLOCK_SIZE];
        for (i, d) in deltas.iter_mut().enumerate() {
            *d = ((i as u32 * 3) % max_delta).max(1);
        }

        // Pack with rounded bitpacking
        let mut packed = vec![0u8; BLOCK_SIZE * rounded_width.bytes_per_value()];
        simd::pack_rounded(&deltas[..], rounded_width, &mut packed);

        // Benchmark: unpack only
        let mut output = [0u32; BLOCK_SIZE];
        let start = Instant::now();
        for _ in 0..iterations {
            simd::unpack_rounded(black_box(&packed), rounded_width, &mut output, BLOCK_SIZE);
            black_box(&output);
        }
        let unpack_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        // Benchmark: unpack + separate delta decode
        let mut doc_ids = [0u32; BLOCK_SIZE];
        let start = Instant::now();
        for block_idx in 0..iterations {
            simd::unpack_rounded(black_box(&packed), rounded_width, &mut output, BLOCK_SIZE);
            let first_doc = block_idx as u32 * 1000;
            simd::delta_decode(&mut doc_ids, &output, first_doc, BLOCK_SIZE);
            black_box(&doc_ids);
        }
        let separate_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        // Benchmark: fused unpack + delta decode
        let start = Instant::now();
        for block_idx in 0..iterations {
            let first_doc = block_idx as u32 * 1000;
            simd::unpack_rounded_delta_decode(
                black_box(&packed),
                rounded_width,
                &mut doc_ids,
                first_doc,
                BLOCK_SIZE,
            );
            black_box(&doc_ids);
        }
        let fused_rate = total_ints as f64 / start.elapsed().as_secs_f64() / 1e9;

        results.push((rounded_bits, unpack_rate, separate_rate, fused_rate));
    }

    // Build table
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("Rounded\nBits"),
        Cell::new("Unpack Only\n(Gints/s)"),
        Cell::new("Unpack+Delta\n(separate)"),
        Cell::new("Fused\n(Gints/s)"),
        Cell::new("Fused vs\nSeparate"),
    ]);

    for (bits, unpack, separate, fused) in &results {
        let speedup = fused / separate;
        let speedup_cell = if speedup > 1.1 {
            Cell::new(format!("{:.2}x", speedup)).fg(Color::Green)
        } else {
            Cell::new(format!("{:.2}x", speedup))
        };

        table.add_row(vec![
            Cell::new(format!("{}", bits)),
            Cell::new(format!("{:.2}", unpack)),
            Cell::new(format!("{:.2}", separate)),
            Cell::new(format!("{:.2}", fused)),
            speedup_cell,
        ]);
    }

    println!("\n📊 FUSED vs SEPARATE ROUNDED DECODE");
    println!(
        "Block size: {} integers | Iterations: {}K blocks\n",
        BLOCK_SIZE,
        iterations / 1000
    );
    println!("{table}");
}
