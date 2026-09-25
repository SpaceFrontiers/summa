//! Adapter for https://github.com/quickwit-oss/search-benchmark-game.
//! See docs/search-benchmark-game.md. Stdout is exclusively the query protocol.
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Arc;
use summa_core::directories::MmapDirectory;
use summa_core::dsl::{PositionMode, QueryLanguageParser};
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::{Collector, CountCollector, Query, TopKCollector, collect_segment};
use summa_core::structures::PostingCodec;
use summa_core::{Document, SchemaBuilder};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Options {
    config: IndexConfig,
    exhaustive: bool,
    background_merges: bool,
    reorder_text: bool,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut options = Self {
            config: IndexConfig {
                num_threads: 1,
                num_indexing_threads: 4,
                max_indexing_memory_bytes: 2_000_000_000,
                ..Default::default()
            },
            exhaustive: false,
            background_merges: true,
            reorder_text: false,
        };
        let mut args = args.iter();
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--posting-ratio-bounds" => options.config.posting_ratio_bounds = true,
                // Impact bounds imply ratio bounds (IndexConfig::effective_posting_bounds).
                "--posting-impact-bounds" => options.config.posting_impact_bounds = true,
                "--no-background-merges" => {
                    options.config.merge_policy = Box::new(summa_core::merge::NoMergePolicy);
                    options.background_merges = false;
                }
                "--exhaustive" => options.exhaustive = true,
                // Capacity limit (0..=65536 blocks) is enforced by IndexConfig at open.
                "--term-cache-blocks" => {
                    options.config.term_cache_blocks =
                        args.next().ok_or("missing term cache capacity")?.parse()?;
                }
                "--term-cache-bytes" => {
                    options.config.term_cache_budget_bytes = Some(
                        args.next()
                            .ok_or("missing term cache byte budget")?
                            .parse()?,
                    );
                }
                "--term-dict-block-bytes" => {
                    options.config.term_dict_block_size = args
                        .next()
                        .ok_or("missing dictionary block target")?
                        .parse()?;
                }
                "--reorder-text" => options.reorder_text = true,
                "--quantized-norms" => options.config.quantized_norms = true,
                "--compact-text" => options.config.compact_text = true,
                "--posting-codec" => {
                    let value = args.next().ok_or("missing posting codec")?;
                    options.config.posting_codec = Some(
                        PostingCodec::parse(value)
                            .ok_or("posting codec must be rounded, packed, pfor, or simd4x")?,
                    );
                }
                "--indexing-threads" => {
                    let value: usize = args
                        .next()
                        .ok_or("missing indexing thread count")?
                        .parse()?;
                    if !(1..=256).contains(&value) {
                        return Err("indexing threads must be in 1..=256".into());
                    }
                    options.config.num_indexing_threads = value;
                }
                "--indexing-memory-bytes" => {
                    let value: usize = args
                        .next()
                        .ok_or("missing indexing memory budget")?
                        .parse()?;
                    if value < 16 * 1024 * 1024 {
                        return Err("indexing memory must be at least 16 MiB".into());
                    }
                    options.config.max_indexing_memory_bytes = value;
                }
                _ => return Err(format!("unknown option: {flag}").into()),
            }
        }
        Ok(options)
    }
}

// Protocol limits are checked before invoking the shared native query parser.
fn parse_query(parser: &QueryLanguageParser, text: &str) -> Result<Box<dyn Query>> {
    if text.len() > 4096 {
        return Err("query exceeds 4096 bytes".into());
    }
    if text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .count()
        > summa_core::query::MAX_QUERY_TERMS
    {
        return Err("query exceeds the core term limit".into());
    }
    parser.parse_strict(text).map_err(Into::into)
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusDoc {
    id: String,
    text: String,
    sort_field: u64,
}

async fn build(path: &Path, options: Options) -> Result<()> {
    if path.exists() {
        return Err("index output already exists; use a new directory".into());
    }
    std::fs::create_dir_all(path)?;
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", false, true);
    let text = schema.add_text_field("text", true, false);
    schema.set_positions(text, PositionMode::TokenPosition);
    schema.set_reorder(text, options.reorder_text);
    let sort = schema.add_u64_field("sort_field", false, false);
    schema.set_fast(sort, true);
    schema.set_default_fields(vec!["text".to_owned()]);
    let mut writer =
        IndexWriter::create(MmapDirectory::new(path), schema.build(), options.config).await?;
    let mut count = 0u64;
    for line in std::io::stdin().lock().lines() {
        let row: CorpusDoc = serde_json::from_str(&line?)?;
        let mut doc = Document::new();
        doc.add_text(id, row.id);
        doc.add_text(text, row.text);
        doc.add_u64(sort, row.sort_field);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        loop {
            match writer.add_document(doc.clone()) {
                Ok(()) => break,
                Err(summa_core::Error::QueueFull) if std::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Err(error) => return Err(error.into()),
            }
        }
        count += 1;
        if count.is_multiple_of(100_000) {
            eprintln!("indexed {count}");
        }
    }
    writer.commit().await?;
    writer.force_merge().await?;
    writer.shutdown().await?;
    eprintln!("indexed {count}; committed and merged");
    Ok(())
}

// Keep core reorder diagnostics off the stdout query protocol.
struct ReorderLogger;
impl log::Log for ReorderLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{} {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

async fn reorder(path: &Path, options: Options) -> Result<()> {
    log::set_logger(&ReorderLogger).map_err(|e| e.to_string())?;
    log::set_max_level(log::LevelFilter::Info);
    eprintln!(
        "bp_memory_budget_bytes={}",
        options.config.bp_memory_budget_bytes
    );
    let mut writer = IndexWriter::open(MmapDirectory::new(path), options.config).await?;
    writer.reorder().await?;
    writer.shutdown().await?;
    Ok(())
}

async fn serve(path: &Path, options: Options) -> Result<()> {
    let index = Index::open(MmapDirectory::new(path), options.config).await?;
    let reader = index.reader().await?;
    let searcher = reader.searcher().await?;
    let parser = searcher.query_parser();
    eprintln!(
        "docs={} segments={}",
        searcher.num_docs(),
        searcher.num_segments()
    );
    if searcher.num_segments() != 1 {
        return Err("benchmark requires exactly one merged segment".into());
    }
    // COUNT/TOP_*_COUNT/VERIFY below run `collect_segment` on the sole segment
    // reader directly, bypassing the searcher's deletion mask. That is exact
    // only because the index was force-merged (no deletions) and the check
    // above refuses multi-segment indexes; `scripts/search_benchmark/smoke.py`
    // verifies the counts against direct token matching.
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let (command, text) = line.split_once('\t').ok_or("expected COMMAND<TAB>query")?;
        let k = match command {
            "COUNT" => 0,
            "TOP_10_COUNT" => 10,
            "TOP_100_COUNT" => 100,
            "TOP_1000_COUNT" => 1000,
            "VERIFY" => 1000,
            "TOP_10" => 10,
            "TOP_100" => 100,
            "TOP_1000" => 1000,
            _ => {
                writeln!(out, "UNSUPPORTED")?;
                out.flush()?;
                continue;
            }
        };
        #[cfg(feature = "query-diagnostics")]
        let parse_started = std::time::Instant::now();
        let query = parse_query(&parser, text)?;
        #[cfg(feature = "query-diagnostics")]
        let parse_ns = parse_started.elapsed().as_nanos();
        #[cfg(feature = "query-diagnostics")]
        let execute_started = std::time::Instant::now();
        let execution = async {
            if command == "COUNT" {
                let mut count = CountCollector::new();
                collect_segment(&searcher.segment_readers()[0], query.as_ref(), &mut count).await?;
                writeln!(out, "{}", count.count())?;
            } else if command.ends_with("_COUNT") || command == "VERIFY" || options.exhaustive {
                let mut count = CountCollector::new();
                let mut top = TopKCollector::new(k);
                if command == "VERIFY" || options.exhaustive {
                    collect_segment(
                        &searcher.segment_readers()[0],
                        query.as_ref(),
                        &mut Exhaustive((&mut top, &mut count)),
                    )
                    .await?;
                } else {
                    collect_segment(
                        &searcher.segment_readers()[0],
                        query.as_ref(),
                        &mut (&mut top, &mut count),
                    )
                    .await?;
                }
                if command == "VERIFY" {
                    let expected = top.into_sorted_results();
                    for limit in [10, 100, 1000] {
                        let (actual, _) =
                            searcher.search_with_offset_and_count_sync(query.as_ref(), limit, 0)?;
                        let wanted = &expected[..expected.len().min(limit)];
                        let mut exact_count = CountCollector::new();
                        let mut exact_top = TopKCollector::new(limit);
                        collect_segment(
                            &searcher.segment_readers()[0],
                            query.as_ref(),
                            &mut (&mut exact_top, &mut exact_count),
                        )
                        .await?;
                        let exact_hits = exact_top.into_sorted_results();
                        if exact_count.count() != count.count()
                            || exact_hits.len() != wanted.len()
                            || exact_hits.iter().zip(wanted).any(|(a, b)| {
                                a.doc_id != b.doc_id || a.score.to_bits() != b.score.to_bits()
                            })
                        {
                            return Err(format!(
                                "exact-count top-{limit} mismatch: {text}; counts {}/{}",
                                exact_count.count(),
                                count.count()
                            )
                            .into());
                        }
                        if actual.len() != wanted.len()
                            || actual.iter().zip(wanted).any(|(a, b)| {
                                a.doc_id != b.doc_id || a.score.to_bits() != b.score.to_bits()
                            })
                        {
                            let first =
                                actual.iter().zip(wanted).enumerate().find(|(_, (a, b))| {
                                    a.doc_id != b.doc_id || a.score.to_bits() != b.score.to_bits()
                                });
                            return Err(format!(
                            "pruned/exhaustive top-{limit} ranking mismatch: {text}; lengths {}/{}; first difference {first:?}", actual.len(), wanted.len()
                        ).into());
                        }
                    }
                    writeln!(out, "{}", count.count())?;
                } else {
                    std::hint::black_box(top.into_sorted_results());
                    writeln!(
                        out,
                        "{}",
                        if command.ends_with("_COUNT") {
                            count.count()
                        } else {
                            1
                        }
                    )?;
                }
            } else {
                let results = searcher.search_with_offset_and_count_sync(query.as_ref(), k, 0)?;
                std::hint::black_box(results);
                writeln!(out, "1")?;
            }

            Ok::<(), Box<dyn std::error::Error>>(())
        };
        #[cfg(feature = "query-diagnostics")]
        {
            let (result, work) = summa_core::search_diagnostics::capture(execution).await;
            let execute_ns = execute_started.elapsed().as_nanos();
            result?;
            eprintln!(
                "QUERY_WORK\t{}",
                serde_json::json!({
                    "schema_version": 1, "command": command, "query": text,
                    "parse_ns": parse_ns, "execute_ns": execute_ns, "work": work,
                })
            );
        }
        #[cfg(not(feature = "query-diagnostics"))]
        execution.await?;
        out.flush()?;
    }
    Ok(())
}

// Verification must score all matches independently of ranked execution.
struct Exhaustive<C>(C);
impl<C: Collector> Collector for Exhaustive<C> {
    fn collect(
        &mut self,
        doc: u32,
        score: f32,
        positions: &[(u32, Vec<summa_core::query::ScoredPosition>)],
    ) {
        self.0.collect(doc, score, positions);
    }
    fn needs_scores(&self) -> bool {
        self.0.needs_scores()
    }
    fn needs_positions(&self) -> bool {
        self.0.needs_positions()
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: search_benchmark_game <index|reorder|serve|validate-queries> <path> [--exhaustive] [--indexing-threads N] [--indexing-memory-bytes N] [--posting-codec rounded|packed|pfor|simd4x] [--compact-text] [--quantized-norms] [--reorder-text] [--term-cache-blocks N] [--term-cache-bytes N] [--term-dict-block-bytes N] [--posting-ratio-bounds] [--posting-impact-bounds] [--no-background-merges]".into());
    }
    let options = Options::parse(&args[3..])?;
    if options.reorder_text && args[1] != "index" {
        return Err(
            "--reorder-text applies only to new indexes; reorder reads persisted eligibility"
                .into(),
        );
    }
    eprintln!(
        "indexing_threads={} indexing_memory_bytes={} posting_codec={} term_cache_blocks={} term_cache_budget_bytes={:?} term_dict_block_bytes={} exhaustive={} posting_ratio_bounds={} posting_impact_bounds={} background_merges={} compact_text={} quantized_norms={} reorder_text={}",
        options.config.num_indexing_threads,
        options.config.max_indexing_memory_bytes,
        options.config.effective_posting_codec(),
        options.config.term_cache_blocks,
        options.config.term_cache_budget_bytes,
        options.config.term_dict_block_size,
        options.exhaustive,
        options.config.effective_posting_bounds().ratio,
        options.config.effective_posting_bounds().impact,
        options.background_merges,
        options.config.compact_text,
        options.config.quantized_norms,
        options.reorder_text
    );
    match args[1].as_str() {
        "index" => build(Path::new(&args[2]), options).await,
        "reorder" => reorder(Path::new(&args[2]), options).await,
        "serve" => serve(Path::new(&args[2]), options).await,
        "validate-queries" => {
            let mut schema = SchemaBuilder::default();
            let field = schema.add_text_field("text", true, false);
            schema.set_positions(field, PositionMode::TokenPosition);
            let parser = QueryLanguageParser::new(
                Arc::new(schema.build()),
                vec![field],
                Arc::new(summa_core::tokenizer::TokenizerRegistry::new()),
            );
            let mut count = 0;
            for line in BufReader::new(std::fs::File::open(&args[2])?).lines() {
                let row: serde_json::Value = serde_json::from_str(&line?)?;
                let text = row
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or("missing query string")?;
                std::hint::black_box(parse_query(&parser, text)?);
                count += 1;
            }
            eprintln!("validated {count} queries");
            Ok(())
        }
        _ => Err("unknown operation".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removed_validation_cache_option_is_not_silently_ignored() {
        assert!(
            Options::parse(&["--posting-validation-cache-bytes".into(), "262144".into()]).is_err()
        );
    }
}
