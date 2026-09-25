//! Executes the pure statistical checks shared with the harness-less memory
//! reserve benchmark. Keeping the arithmetic outside device code makes ratio
//! and acceptance reporting part of the ordinary test suite.

#[path = "../benches/memory_reserve/fairness.rs"]
mod fairness;
#[path = "../benches/memory_reserve/stats.rs"]
mod stats;

use stats::{ExecutionOrder, PairedSample, PairedTrial};

fn sample(reference: f64, candidate: f64) -> PairedSample {
    PairedSample::new(
        0,
        ExecutionOrder::ReferenceThenCandidate,
        reference,
        candidate,
    )
    .unwrap()
}

fn production_pair() -> (summa_llm::ModelDef, summa_llm::ModelDef) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../summa-mal/well-known");
    (
        summa_llm::parse_mal_file(root.join("retriever_300m_moe.mal")).unwrap(),
        summa_llm::parse_mal_file(root.join("retriever_300m_moe_sleep.mal")).unwrap(),
    )
}

fn first_memory_tiers(model: &mut summa_llm::ModelDef) -> &mut [summa_llm::MemoryTierDef] {
    &mut model.pattern.as_mut().unwrap()[0]
        .memory
        .as_mut()
        .unwrap()
        .tiers
}

#[test]
fn paired_summary_uses_ratios_of_matched_samples() {
    let trials = vec![
        PairedTrial::new(1, vec![sample(10.0, 11.0), sample(20.0, 18.0)]).unwrap(),
        PairedTrial::new(2, vec![sample(100.0, 105.0), sample(80.0, 88.0)]).unwrap(),
        PairedTrial::new(3, vec![sample(50.0, 50.0), sample(40.0, 44.0)]).unwrap(),
    ];
    let summary = stats::summarize(&trials).unwrap();
    assert_eq!(summary.seed_count, 3);
    assert_eq!(summary.sample_count, 6);
    assert!((summary.paired_ratio_median - 1.075).abs() < 1e-12);
    assert!((summary.median_of_seed_median_ratios - 1.05).abs() < 1e-12);
    assert!((summary.worst_seed_median_ratio - 1.075).abs() < 1e-12);
}

#[test]
fn invalid_timing_is_rejected_before_json_output() {
    assert!(PairedSample::new(0, ExecutionOrder::CandidateThenReference, 0.0, 1.0).is_err());
    assert!(PairedSample::new(0, ExecutionOrder::CandidateThenReference, 1.0, f64::NAN).is_err());
    assert!(
        PairedSample::new(
            0,
            ExecutionOrder::CandidateThenReference,
            f64::MIN_POSITIVE,
            f64::MAX,
        )
        .is_err()
    );
    assert!(stats::summarize(&[]).is_err());
}

#[test]
fn production_static_and_sleep_models_are_a_matched_pair() {
    let (static_model, memory_model) = production_pair();
    fairness::validate_matched_backbone(&static_model, &memory_model).unwrap();
}

#[test]
fn fairness_check_rejects_an_easier_candidate() {
    let (static_model, mut memory_model) = production_pair();
    memory_model.hidden_size /= 2;
    assert!(fairness::validate_matched_backbone(&static_model, &memory_model).is_err());
}

#[test]
fn fairness_check_requires_later_tiers_to_start_as_noops() {
    let (static_model, mut memory_model) = production_pair();
    first_memory_tiers(&mut memory_model)[1].residual_init = summa_llm::MemoryTierInit::Default;
    assert!(fairness::validate_matched_backbone(&static_model, &memory_model).is_err());
}

#[test]
fn fairness_check_requires_fast_tier_to_preserve_static_ffn() {
    let (static_model, mut memory_model) = production_pair();
    first_memory_tiers(&mut memory_model)[0].residual_init =
        summa_llm::MemoryTierInit::ResidualZero;
    assert!(fairness::validate_matched_backbone(&static_model, &memory_model).is_err());
}
