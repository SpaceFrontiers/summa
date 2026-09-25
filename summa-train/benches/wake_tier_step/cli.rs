//! Command-line parsing and bounded workload validation.

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};

const MAX_BATCH_SIZE: usize = 65_536;
const MAX_SEQUENCE_LENGTH: usize = 8_192;
const MAX_LOGICAL_TOKENS: usize = 16 * 1024 * 1024;
const MAX_WARMUPS: usize = 100;
const MAX_ITERATIONS: usize = 10_000;
const MAX_SEEDS: usize = 64;
const MAX_PAIRED_SAMPLES: usize = 100_000;
const MAX_TIERS: usize = 32;

#[derive(Debug)]
pub(super) struct Args {
    pub(super) model: Option<PathBuf>,
    pub(super) output: Option<PathBuf>,
    pub(super) batch_size: usize,
    pub(super) sequence_length: usize,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) seeds: Vec<u64>,
    pub(super) update_periods: Vec<u64>,
    pub(super) non_due_clock: u64,
    pub(super) due_clock: u64,
    pub(super) require_cuda: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: None,
            output: None,
            batch_size: 2,
            sequence_length: 8,
            warmup: 3,
            iterations: 10,
            seeds: vec![17, 23, 29],
            update_periods: vec![2, 4],
            non_due_clock: 1,
            due_clock: 2,
            require_cuda: false,
        }
    }
}

pub(super) fn parse_args() -> Result<Args> {
    parse_args_from(std::env::args().skip(1))
}

pub(super) fn parse_args_from(mut values: impl Iterator<Item = String>) -> Result<Args> {
    let mut parsed = Args::default();
    while let Some(flag) = values.next() {
        let mut value = || {
            values
                .next()
                .with_context(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--model" => parsed.model = Some(PathBuf::from(value()?)),
            "--output" => parsed.output = Some(PathBuf::from(value()?)),
            "--batch-size" => {
                parsed.batch_size = value()?.parse().context("invalid --batch-size")?
            }
            "--sequence-length" => {
                parsed.sequence_length = value()?.parse().context("invalid --sequence-length")?
            }
            "--warmup" => parsed.warmup = value()?.parse().context("invalid --warmup")?,
            "--iterations" => {
                parsed.iterations = value()?.parse().context("invalid --iterations")?
            }
            "--seeds" => parsed.seeds = parse_csv(&value()?, "--seeds")?,
            "--periods" => parsed.update_periods = parse_csv(&value()?, "--periods")?,
            "--non-due-clock" => {
                parsed.non_due_clock = value()?.parse().context("invalid --non-due-clock")?
            }
            "--due-clock" => parsed.due_clock = value()?.parse().context("invalid --due-clock")?,
            "--require-cuda" => parsed.require_cuda = true,
            // Cargo appends this marker for harness-less benchmark targets.
            "--bench" => {}
            "--help" | "-h" => {
                println!(
                    "Usage: wake_tier_step [--model PATH] [--batch-size N] [--sequence-length N] [--warmup N] [--iterations N] [--seeds A,B,C] [--periods FAST,SLOW,...] [--non-due-clock N] [--due-clock N] [--require-cuda] [--output PATH]"
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument {flag:?}"),
        }
    }
    validate(parsed)
}

pub(super) fn validate(parsed: Args) -> Result<Args> {
    ensure!(
        (1..=MAX_BATCH_SIZE).contains(&parsed.batch_size),
        "--batch-size must be in 1..={MAX_BATCH_SIZE}"
    );
    ensure!(
        (2..=MAX_SEQUENCE_LENGTH).contains(&parsed.sequence_length),
        "--sequence-length must be in 2..={MAX_SEQUENCE_LENGTH}"
    );
    let logical_tokens = parsed
        .batch_size
        .checked_mul(parsed.sequence_length)
        .context("benchmark token count overflows usize")?;
    ensure!(
        logical_tokens <= MAX_LOGICAL_TOKENS,
        "benchmark workload exceeds the {MAX_LOGICAL_TOKENS}-token safety bound"
    );
    ensure!(
        parsed.warmup <= MAX_WARMUPS,
        "--warmup must not exceed {MAX_WARMUPS}"
    );
    ensure!(
        (1..=MAX_ITERATIONS).contains(&parsed.iterations),
        "--iterations must be in 1..={MAX_ITERATIONS}"
    );
    ensure!(
        (3..=MAX_SEEDS).contains(&parsed.seeds.len()),
        "--seeds must contain between 3 and {MAX_SEEDS} values"
    );
    ensure!(
        parsed.seeds.windows(2).all(|pair| pair[0] < pair[1]),
        "paired seeds must be unique and strictly increasing"
    );
    let paired_samples = parsed
        .seeds
        .len()
        .checked_mul(parsed.iterations)
        .context("paired sample count overflows usize")?;
    ensure!(
        paired_samples <= MAX_PAIRED_SAMPLES,
        "benchmark run exceeds the {MAX_PAIRED_SAMPLES}-sample safety bound"
    );
    ensure!(
        (2..=MAX_TIERS).contains(&parsed.update_periods.len()),
        "--periods must contain between 2 and {MAX_TIERS} tier periods"
    );
    ensure!(
        parsed.update_periods.iter().all(|period| *period > 0),
        "--periods values must be positive"
    );
    ensure!(
        parsed.non_due_clock > 0 && parsed.due_clock > 0,
        "benchmark clocks must be positive"
    );
    ensure!(
        parsed
            .update_periods
            .iter()
            .all(|period| !parsed.non_due_clock.is_multiple_of(*period)),
        "--non-due-clock is a boundary for at least one configured tier"
    );
    ensure!(
        parsed
            .update_periods
            .iter()
            .any(|period| parsed.due_clock.is_multiple_of(*period)),
        "--due-clock is not a boundary for any configured tier"
    );
    Ok(parsed)
}

fn parse_csv<T>(value: &str, flag: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value
        .split(',')
        .map(|item| {
            ensure!(!item.is_empty(), "{flag} contains an empty value");
            item.parse()
                .with_context(|| format!("invalid value in {flag}"))
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
