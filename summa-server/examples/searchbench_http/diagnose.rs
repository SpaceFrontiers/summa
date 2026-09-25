//! Read-only stage timings and optional work counters through existing owners.
use std::hint::black_box;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use serde_json::{Value, json};
use summa_core::IndexConfig;

pub async fn run(path: &Path, input: &Path, config: IndexConfig) -> Result<()> {
    let app = super::open_app(path, config).await?;
    let source = std::io::BufReader::new(std::fs::File::open(input)?);
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in source.lines() {
        let value: Value = serde_json::from_str(&line?)?;
        for limit in [10, 100] {
            let mut envelope = value.clone();
            envelope["limit"] = json!(limit);
            let bytes = serde_json::to_vec(&envelope)?;
            let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut response_bytes = 0;
            let mut returned = 0;
            for iteration in 0..120 {
                let start = Instant::now();
                let request = super::query::Envelope::from_value(serde_json::from_slice(&bytes)?)?
                    .parse(&app.parser, app.body, &app.tokenizer)?;
                let parsed = Instant::now();
                let (hits, _) = app.searcher.search_with_offset_and_count_sync(
                    request.query.as_ref(),
                    limit,
                    0,
                )?;
                let searched = Instant::now();
                let response = app.project(&hits)?;
                let projected = Instant::now();
                let serialized = serde_json::to_vec(&response)?;
                let encoded = Instant::now();
                drop(response);
                let finished = Instant::now();
                returned = hits.len();
                response_bytes = serialized.len();
                black_box(serialized);
                if iteration >= 20 {
                    for (samples, elapsed) in samples.iter_mut().zip([
                        parsed.duration_since(start),
                        searched.duration_since(parsed),
                        projected.duration_since(searched),
                        encoded.duration_since(projected),
                        finished.duration_since(encoded),
                    ]) {
                        samples.push(elapsed.as_nanos() as u64);
                    }
                }
            }
            let medians: Vec<_> = samples
                .iter_mut()
                .map(|values| {
                    values.sort_unstable();
                    (values[49] as f64 + values[50] as f64) / 2.0 / 1000.0
                })
                .collect();
            #[cfg(feature = "query-diagnostics")]
            let work = {
                let request = super::query::Envelope::from_value(envelope)?.parse(
                    &app.parser,
                    app.body,
                    &app.tokenizer,
                )?;
                let (result, work) = summa_core::search_diagnostics::capture_sync(|| {
                    app.searcher
                        .search_with_offset_and_count_sync(request.query.as_ref(), limit, 0)
                });
                black_box(result?);
                serde_json::to_value(work)?
            };
            #[cfg(not(feature = "query-diagnostics"))]
            let work = Value::Null;
            writeln!(
                output,
                "{}",
                json!({
                    "query":value["query"], "class":value["class"], "limit":limit,
                    "samples":100, "warmup":20, "returned":returned,
                    "response_bytes":response_bytes, "instrumented":cfg!(feature="query-diagnostics"),
                    "median_us":{"parse":medians[0],"search":medians[1],"project":medians[2],"serialize":medians[3],"response_drop":medians[4]},
                    "work":work,
                })
            )?;
            output.flush()?;
        }
    }
    Ok(())
}
