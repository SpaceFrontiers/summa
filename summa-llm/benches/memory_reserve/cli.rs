//! Command-line parsing and acceptance-run guardrails.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};

#[derive(Debug)]
pub(super) struct Args {
    pub(super) model: Option<PathBuf>,
    pub(super) baseline_model: Option<PathBuf>,
    pub(super) output: Option<PathBuf>,
    pub(super) tokens: usize,
    pub(super) tier: usize,
    pub(super) max_active: usize,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) seeds: Vec<u64>,
    pub(super) max_overhead_percent: f64,
    pub(super) enforce: bool,
    pub(super) require_cuda: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: None,
            baseline_model: None,
            output: None,
            tokens: 4_096,
            tier: 0,
            max_active: 2,
            warmup: 3,
            iterations: 10,
            seeds: vec![17, 23, 29],
            max_overhead_percent: 5.0,
            enforce: false,
            require_cuda: false,
        }
    }
}

pub(super) fn parse_args() -> Result<Args> {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from(mut values: impl Iterator<Item = String>) -> Result<Args> {
    let mut parsed = Args::default();
    while let Some(flag) = values.next() {
        let mut value = || {
            values
                .next()
                .with_context(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--model" => parsed.model = Some(PathBuf::from(value()?)),
            "--baseline-model" => parsed.baseline_model = Some(PathBuf::from(value()?)),
            "--output" => parsed.output = Some(PathBuf::from(value()?)),
            "--tokens" => parsed.tokens = value()?.parse().context("invalid --tokens")?,
            "--tier" => parsed.tier = value()?.parse().context("invalid --tier")?,
            "--max-active" => {
                parsed.max_active = value()?.parse().context("invalid --max-active")?
            }
            "--warmup" => parsed.warmup = value()?.parse().context("invalid --warmup")?,
            "--iterations" => {
                parsed.iterations = value()?.parse().context("invalid --iterations")?
            }
            "--seeds" => parsed.seeds = parse_seeds(&value()?)?,
            "--max-overhead-percent" => {
                parsed.max_overhead_percent =
                    value()?.parse().context("invalid --max-overhead-percent")?
            }
            "--enforce" => parsed.enforce = true,
            "--require-cuda" => parsed.require_cuda = true,
            // Cargo appends this marker for harness-less bench targets.
            "--bench" => {}
            "--help" | "-h" => {
                println!(
                    "Usage: memory_reserve [--model PATH --baseline-model PATH] [--tokens N] [--tier N] [--max-active N] [--warmup N] [--iterations N] [--seeds A,B,C] [--max-overhead-percent P] [--enforce] [--require-cuda] [--output PATH]"
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument {flag:?}"),
        }
    }
    validate(parsed)
}

fn validate(parsed: Args) -> Result<Args> {
    ensure!(
        parsed.model.is_some() == parsed.baseline_model.is_some(),
        "--model and --baseline-model must be supplied together"
    );
    ensure!(
        parsed.tokens > 0 && parsed.iterations > 0 && parsed.max_active > 0,
        "--tokens, --iterations, and --max-active must be positive"
    );
    ensure!(
        parsed.seeds.len() >= 3,
        "at least three paired seeds are required"
    );
    ensure!(
        parsed.seeds.iter().copied().collect::<BTreeSet<_>>().len() == parsed.seeds.len(),
        "paired seeds must be unique"
    );
    ensure!(
        parsed.max_overhead_percent.is_finite() && parsed.max_overhead_percent >= 0.0,
        "--max-overhead-percent must be finite and non-negative"
    );
    if parsed.enforce {
        ensure!(
            parsed.warmup >= 3 && parsed.iterations >= 10,
            "--enforce requires at least 3 warmups and 10 measured iterations per seed"
        );
        ensure!(
            parsed.max_active >= 2,
            "--enforce requires --max-active of at least 2 to test constant active compute"
        );
    }
    Ok(parsed)
}

fn parse_seeds(value: &str) -> Result<Vec<u64>> {
    value
        .split(',')
        .map(|seed| {
            ensure!(!seed.is_empty(), "--seeds contains an empty value");
            seed.parse().context("invalid seed in --seeds")
        })
        .collect()
}

pub(super) fn backend_name() -> &'static str {
    if cfg!(all(
        feature = "cuda",
        target_os = "linux",
        not(feature = "metal")
    )) {
        "cuda"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "ndarray"
    }
}
