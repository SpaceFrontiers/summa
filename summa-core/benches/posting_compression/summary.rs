//! Summary table benchmarks with formatted output

use comfy_table::{Cell, Color, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use criterion::Criterion;

use super::access_patterns::{AccessPattern, measure_access_pattern};
use super::block_decompression::bench_raw_throughput;
use super::common::{
    Distribution, FORMAT_SHORT, FORMAT_SHORT_7, find_best_idx, find_min_idx, format_rate,
    generate_postings, measure_compression,
};
use super::encoding::measure_encoding_speed;
use super::iteration::measure_decoding_speed;

/// Comprehensive benchmark summary with formatted tables
pub fn bench_all_formats_compression_summary(_c: &mut Criterion) {
    println!("\n🚀 POSTING LIST COMPRESSION BENCHMARK RESULTS\n");
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FORMAT LEGEND                                                               │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ BP128   - Bitpacking with 128-int blocks, exact bit widths                  │");
    println!("│ BP-Rnd  - Bitpacking with rounded bit widths (8/16/32) for faster SIMD      │");
    println!("│ VBP128  - Vertical bitpacking (SIMD-optimized bit-interleaved layout)       │");
    println!("│ EF      - Elias-Fano encoding for monotone integer sequences                │");
    println!("│ PEF     - Partitioned Elias-Fano (better compression & skip performance)    │");
    println!("│ Roaring - Roaring bitmaps (hybrid: sorted arrays / bitmaps / RLE)           │");
    println!("│ OptP4D  - Optimized PFor-Delta with SIMD (patched frame-of-reference)       │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    // ═══════════════════════════════════════════════════════════════════════════════
    // COMPRESSION RATIO TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut compression_table = Table::new();
    compression_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Distribution",
            "Size",
            "BP128",
            "BP-Rnd",
            "VBP128",
            "EF",
            "PEF",
            "Roaring",
            "OptP4D",
            "Best",
        ]);

    let distributions = [
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
        Distribution::Clustered,
        Distribution::Sequential,
    ];
    let sizes = [1000, 10000, 50000];

    let mut total_ratios = [0.0f64; 7];
    let mut count = 0;

    for dist in &distributions {
        for &size in &sizes {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let result = measure_compression(&doc_ids, &term_freqs);

            let ratios = [
                result.horiz_bp128_ratio(),
                result.horiz_bp128_rounded_ratio(),
                result.vert_bp128_ratio(),
                result.elias_fano_ratio(),
                result.partitioned_ef_ratio(),
                result.roaring_ratio(),
                result.opt_p4d_ratio(),
            ];

            for i in 0..7 {
                total_ratios[i] += ratios[i];
            }
            count += 1;

            let best_idx = find_min_idx(&ratios);

            compression_table.add_row(vec![
                dist.name(),
                &size.to_string(),
                &format!("{:.1}%", ratios[0]),
                &format!("{:.1}%", ratios[1]),
                &format!("{:.1}%", ratios[2]),
                &format!("{:.1}%", ratios[3]),
                &format!("{:.1}%", ratios[4]),
                &format!("{:.1}%", ratios[5]),
                &format!("{:.1}%", ratios[6]),
                FORMAT_SHORT_7[best_idx],
            ]);
        }
    }

    let avg: Vec<f64> = total_ratios.iter().map(|r| r / count as f64).collect();
    let best_avg_idx = find_min_idx(&avg);
    compression_table.add_row(vec![
        "AVERAGE",
        "ALL",
        &format!("{:.1}%", avg[0]),
        &format!("{:.1}%", avg[1]),
        &format!("{:.1}%", avg[2]),
        &format!("{:.1}%", avg[3]),
        &format!("{:.1}%", avg[4]),
        &format!("{:.1}%", avg[5]),
        &format!("{:.1}%", avg[6]),
        FORMAT_SHORT_7[best_avg_idx],
    ]);

    println!("📊 COMPRESSION RATIO (% of raw size - lower is better)");
    println!("{compression_table}");

    // ═══════════════════════════════════════════════════════════════════════════════
    // ENCODING SPEED TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut encoding_table = Table::new();
    encoding_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Distribution",
            "Size",
            "BP128",
            "BP-Rnd",
            "VBP128",
            "EF",
            "PEF",
            "Roaring",
            "OptP4D",
            "Fastest",
        ]);

    let mut total_enc_rates = [0.0f64; 7];
    let mut enc_count = 0;

    for dist in &[
        Distribution::Medium,
        Distribution::Sparse,
        Distribution::Dense,
    ] {
        for &size in &[1000, 10000, 100000] {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let iterations = if size < 5000 {
                500
            } else if size < 50000 {
                100
            } else {
                50
            };
            let rates = measure_encoding_speed(&doc_ids, &term_freqs, iterations);

            for i in 0..7 {
                total_enc_rates[i] += rates[i];
            }
            enc_count += 1;

            let best_idx = find_best_idx(&rates);

            encoding_table.add_row(vec![
                dist.name(),
                &size.to_string(),
                &format_rate(rates[0]),
                &format_rate(rates[1]),
                &format_rate(rates[2]),
                &format_rate(rates[3]),
                &format_rate(rates[4]),
                &format_rate(rates[5]),
                &format_rate(rates[6]),
                FORMAT_SHORT_7[best_idx],
            ]);
        }
    }

    let avg_enc: Vec<f64> = total_enc_rates
        .iter()
        .map(|r| r / enc_count as f64)
        .collect();
    let best_enc_idx = find_best_idx(&avg_enc);
    encoding_table.add_row(vec![
        "AVERAGE",
        "ALL",
        &format_rate(avg_enc[0]),
        &format_rate(avg_enc[1]),
        &format_rate(avg_enc[2]),
        &format_rate(avg_enc[3]),
        &format_rate(avg_enc[4]),
        &format_rate(avg_enc[5]),
        &format_rate(avg_enc[6]),
        FORMAT_SHORT_7[best_enc_idx],
    ]);

    println!("\n⚡ ENCODING SPEED (K=1000 ints/μs, higher is better)");
    println!("{encoding_table}");

    // ═══════════════════════════════════════════════════════════════════════════════
    // DECODING SPEED TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut decoding_table = Table::new();
    decoding_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Distribution",
            "Size",
            "BP128",
            "BP-Rnd",
            "VBP128",
            "EF",
            "PEF",
            "Roaring",
            "OptP4D",
            "Fastest",
        ]);

    let mut total_dec_rates = [0.0f64; 7];
    let mut dec_count = 0;

    for dist in &[
        Distribution::Medium,
        Distribution::Sparse,
        Distribution::Dense,
    ] {
        for &size in &[1000, 10000, 100000] {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let iterations = if size < 5000 {
                2000
            } else if size < 50000 {
                500
            } else {
                100
            };
            let rates = measure_decoding_speed(&doc_ids, &term_freqs, iterations);

            for i in 0..7 {
                total_dec_rates[i] += rates[i];
            }
            dec_count += 1;

            let best_idx = find_best_idx(&rates);

            decoding_table.add_row(vec![
                dist.name(),
                &size.to_string(),
                &format_rate(rates[0]),
                &format_rate(rates[1]),
                &format_rate(rates[2]),
                &format_rate(rates[3]),
                &format_rate(rates[4]),
                &format_rate(rates[5]),
                &format_rate(rates[6]),
                FORMAT_SHORT_7[best_idx],
            ]);
        }
    }

    let avg_dec: Vec<f64> = total_dec_rates
        .iter()
        .map(|r| r / dec_count as f64)
        .collect();
    let best_dec_idx = find_best_idx(&avg_dec);
    decoding_table.add_row(vec![
        "AVERAGE",
        "ALL",
        &format_rate(avg_dec[0]),
        &format_rate(avg_dec[1]),
        &format_rate(avg_dec[2]),
        &format_rate(avg_dec[3]),
        &format_rate(avg_dec[4]),
        &format_rate(avg_dec[5]),
        &format_rate(avg_dec[6]),
        FORMAT_SHORT_7[best_dec_idx],
    ]);

    println!("\n🔄 DECODING SPEED (K=1000 ints/μs, higher is better)");
    println!("{decoding_table}");

    // ═══════════════════════════════════════════════════════════════════════════════
    // ACCESS PATTERNS TABLE
    // ═══════════════════════════════════════════════════════════════════════════════
    let mut access_table = Table::new();
    access_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            "Pattern", "HorizBP", "VertBP", "EF", "PEF", "Roaring", "OptP4D", "Best",
        ]);

    let (access_doc_ids, access_term_freqs) = generate_postings(50_000, Distribution::Medium);
    let patterns = [
        (AccessPattern::FullScan, 100),
        (AccessPattern::TopN(100), 5000),
        (AccessPattern::RandomSeek, 1000),
        (AccessPattern::SkipInterval, 500),
        (AccessPattern::Galloping, 1000),
    ];

    for (pattern, iterations) in &patterns {
        let rates =
            measure_access_pattern(&access_doc_ids, &access_term_freqs, *pattern, *iterations);
        let best_idx = find_best_idx(&rates);

        let mut row: Vec<Cell> = vec![Cell::new(pattern.name())];
        for (i, &rate) in rates.iter().enumerate() {
            let cell = if i == best_idx {
                Cell::new(format_rate(rate)).fg(Color::Green)
            } else {
                Cell::new(format_rate(rate))
            };
            row.push(cell);
        }
        row.push(Cell::new(FORMAT_SHORT[best_idx]));
        access_table.add_row(row);
    }

    println!("\n🎯 ACCESS PATTERNS (50K docs, K=1000 ops/μs, higher is better)");
    println!("{access_table}");

    // Raw throughput table
    bench_raw_throughput(_c);
}
