//! Executable tests for pure support code used by the harness-less benchmark.
//!
//! A `harness = false` target has no libtest entry point of its own. Including
//! only its side-effect-free CLI and statistics modules here keeps their
//! bounds and paired aggregation in the normal summa-train test matrix.

#[allow(dead_code)]
#[path = "../benches/wake_tier_step/cli.rs"]
mod cli;
#[allow(dead_code)]
#[path = "../benches/wake_tier_step/stats.rs"]
mod stats;

fn parse(values: &[&str]) -> anyhow::Result<cli::Args> {
    cli::parse_args_from(values.iter().map(|value| (*value).to_owned()))
}

#[test]
fn defaults_are_a_bounded_three_seed_pair() {
    let args = parse(&[]).unwrap();
    assert_eq!(args.seeds, vec![17, 23, 29]);
    assert_eq!(args.update_periods, vec![2, 4]);
    assert!(
        args.update_periods
            .iter()
            .all(|period| !args.non_due_clock.is_multiple_of(*period))
    );
    assert!(
        args.update_periods
            .iter()
            .any(|period| args.due_clock.is_multiple_of(*period))
    );
}

#[test]
fn rejects_unbounded_or_statistically_invalid_runs() {
    assert!(parse(&["--iterations", "0"]).is_err());
    assert!(parse(&["--iterations", "10001"]).is_err());
    assert!(parse(&["--seeds", "1,2"]).is_err());
    assert!(parse(&["--seeds", "1,1,2"]).is_err());
    assert!(parse(&["--seeds", "1,3,2"]).is_err());
    assert!(parse(&["--batch-size", "65536", "--sequence-length", "8192"]).is_err());
    let too_many_samples = cli::Args {
        seeds: (0..64_u64).collect(),
        iterations: 10_000,
        ..Default::default()
    };
    assert!(cli::validate(too_many_samples).is_err());
}

#[test]
fn rejects_mislabelled_boundary_clocks() {
    assert!(parse(&["--non-due-clock", "2"]).is_err());
    assert!(parse(&["--due-clock", "3"]).is_err());
    assert!(parse(&["--periods", "2"]).is_err());
    assert!(parse(&["--periods", "0,4"]).is_err());
}

#[test]
fn paired_summary_keeps_raw_samples_and_uses_per_seed_ratios() {
    use stats::{ExecutionOrder, PairedSample, PairedTrial};

    let trials = vec![
        PairedTrial::new(
            17,
            vec![
                PairedSample::new(0, ExecutionOrder::NonDueThenDue, 2.0, 4.0).unwrap(),
                PairedSample::new(1, ExecutionOrder::DueThenNonDue, 4.0, 6.0).unwrap(),
            ],
        )
        .unwrap(),
        PairedTrial::new(
            23,
            vec![PairedSample::new(0, ExecutionOrder::DueThenNonDue, 5.0, 5.0).unwrap()],
        )
        .unwrap(),
        PairedTrial::new(
            29,
            vec![PairedSample::new(0, ExecutionOrder::NonDueThenDue, 3.0, 9.0).unwrap()],
        )
        .unwrap(),
    ];

    let summary = stats::summarize(&trials).unwrap();

    assert_eq!(summary.seed_count, 3);
    assert_eq!(summary.sample_count, 4);
    assert_eq!(summary.non_due_median_ms, 3.5);
    assert_eq!(summary.due_median_ms, 5.5);
    assert_eq!(summary.paired_ratio_median, 1.75);
    assert_eq!(summary.median_of_seed_median_ratios, 1.75);
    assert_eq!(summary.worst_seed_median_ratio, 3.0);
    assert_eq!(trials[0].samples.len(), 2);
}

#[test]
fn durations_must_be_finite_and_positive() {
    use stats::{ExecutionOrder, PairedSample};

    assert!(PairedSample::new(0, ExecutionOrder::NonDueThenDue, 0.0, 1.0).is_err());
    assert!(PairedSample::new(0, ExecutionOrder::NonDueThenDue, 1.0, f64::NAN).is_err());
}
