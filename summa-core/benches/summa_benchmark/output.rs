//! Unified output formatting with comfy_table

use comfy_table::{Cell, Color, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};

/// Create a styled table with standard formatting
pub fn create_table(headers: Vec<&str>) -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(headers.iter().map(|h| Cell::new(*h).fg(Color::Cyan)));
    table
}

/// Format duration as human-readable time
pub fn format_duration(secs: f64) -> String {
    if secs >= 60.0 {
        format!("{:.1}m", secs / 60.0)
    } else if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.1}ms", secs * 1000.0)
    }
}

/// Format latency in microseconds
pub fn format_latency(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:.2}ms", us / 1000.0)
    } else {
        format!("{:.1}μs", us)
    }
}

/// Format percentage with color based on value
pub fn format_recall(value: f32, threshold_green: f32, threshold_yellow: f32) -> Cell {
    let text = format!("{:.1}%", value * 100.0);
    if value >= threshold_green {
        Cell::new(text).fg(Color::Green)
    } else if value >= threshold_yellow {
        Cell::new(text).fg(Color::Yellow)
    } else {
        Cell::new(text).fg(Color::Red)
    }
}

/// Format percentage difference
pub fn format_diff(value: f32, baseline: f32) -> String {
    if (value - baseline).abs() < 0.001 {
        "baseline".to_string()
    } else {
        let diff = (value / baseline - 1.0) * 100.0;
        format!("{:+.1}%", diff)
    }
}

/// Format speedup multiplier
pub fn format_speedup(value: f64) -> String {
    format!("{:.2}x", value)
}

/// Print section header
pub fn print_section(title: &str) {
    println!("\n┌{}┐", "─".repeat(77));
    println!("│ {:<75} │", title);
    println!("└{}┘", "─".repeat(77));
}

/// Print benchmark suite header
pub fn print_header(dataset_info: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                        SUMMA BENCHMARK SUITE");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("{}", dataset_info);
    println!();
}

/// MRL comparison result
pub struct MrlResult {
    pub dim: usize,
    pub recall_at_10: f32,
    pub latency_us: f64,
}

/// Print MRL dimension comparison table
pub fn print_mrl_table(results: &[MrlResult], full_dim: usize) {
    print_section("1. MATRYOSHKA DIMENSION COMPARISON");

    let mut table = create_table(vec!["Dim", "Recall@10", "vs Full", "Latency", "Speedup"]);

    let full_result = results
        .iter()
        .find(|r| r.dim == full_dim)
        .or(results.last());
    let (full_recall, full_latency) = full_result
        .map(|r| (r.recall_at_10, r.latency_us))
        .unwrap_or((1.0, 1.0));

    for r in results {
        let speedup = full_latency / r.latency_us;
        table.add_row(vec![
            Cell::new(r.dim.to_string()),
            format_recall(r.recall_at_10, 0.90, 0.80),
            Cell::new(format_diff(r.recall_at_10, full_recall)),
            Cell::new(format_latency(r.latency_us)),
            Cell::new(format_speedup(speedup)),
        ]);
    }

    println!("{table}");
}

/// Print nprobe comparison table (reuses MrlResult where dim = nprobe)
pub fn print_nprobe_table(results: &[MrlResult], mrl_dim: usize) {
    print_section(&format!("2. NPROBE COMPARISON (mrl_dim={})", mrl_dim));

    let mut table = create_table(vec!["nprobe", "Recall@10", "Latency", "Speedup"]);

    let baseline_latency = results.first().map(|r| r.latency_us).unwrap_or(1.0);

    for r in results {
        let speedup = baseline_latency / r.latency_us;
        table.add_row(vec![
            Cell::new(r.dim.to_string()), // dim field holds nprobe value
            format_recall(r.recall_at_10, 0.90, 0.80),
            Cell::new(format_latency(r.latency_us)),
            Cell::new(format_speedup(speedup)),
        ]);
    }

    println!("{table}");
}

/// Indexing performance result
pub struct IndexingResult {
    pub name: String,
    pub build_time_secs: f64,
    pub merge_time_secs: Option<f64>,
    pub throughput_docs_per_sec: f64,
}

/// Print indexing performance table
pub fn print_indexing_table(results: &[IndexingResult]) {
    print_section("5. INDEXING PERFORMANCE");

    let mut table = create_table(vec![
        "Index Type",
        "Build Time",
        "Merge Time",
        "Throughput (docs/sec)",
    ]);

    for r in results {
        table.add_row(vec![
            Cell::new(&r.name),
            Cell::new(format_duration(r.build_time_secs)),
            Cell::new(r.merge_time_secs.map_or("-".to_string(), format_duration)),
            Cell::new(format!("{:.0}", r.throughput_docs_per_sec)),
        ]);
    }

    println!("{table}");
}

/// IR metrics result (for MS MARCO evaluation)
pub struct IrResult {
    pub name: String,
    pub mrr_at_10: f32,
    pub ndcg_at_10: f32,
    pub recall_at_100: f32,
    pub latency_us: f64,
}

/// Print IR metrics comparison table
pub fn print_ir_table(results: &[IrResult]) {
    print_section("4. MS MARCO IR METRICS (using qrels)");

    let mut table = create_table(vec!["Method", "MRR@10", "NDCG@10", "Recall@100", "Latency"]);

    for r in results {
        table.add_row(vec![
            Cell::new(&r.name),
            Cell::new(format!("{:.4}", r.mrr_at_10)),
            Cell::new(format!("{:.4}", r.ndcg_at_10)),
            format_recall(r.recall_at_100, 0.90, 0.80),
            Cell::new(format_latency(r.latency_us)),
        ]);
    }

    println!("{table}");
}
