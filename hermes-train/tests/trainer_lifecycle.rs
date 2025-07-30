use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use burn::tensor::{Int, Tensor};
use burn_optim::GradientsParams;
use hermes_llm::{Device, Transformer, parse_mal};
use hermes_train::metrics::{MetricEvent, MetricRecord, QuantizationStage};
use hermes_train::native_sleep::{NativeCheckpointRef, NativeSleepCheckpoint};
use hermes_train::qat_candidate::open_qat_candidate;
use hermes_train::quantization::{UltraQuantFormat, fake_quantized_transformer};
use hermes_train::runtime::workflow_signature;
use hermes_train::sleep::{
    ImitationConfig, KnowledgeSeedingConfig, MemoryTierSchedule, SleepSchedule,
    TerminalConsolidation, UpdateClock,
};
use hermes_train::tensor_sleep::{RetentionGateConfig, TensorTransactionStore};
use hermes_train::tier_optimizer::{
    DurableTierOptimizerPublisher, TierOptimizerBank, TierOptimizerConfig,
};
use hermes_train::workflow::{InModelSleepConfig, load_workflow};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const ORDINARY_MODEL: &str = r#"
ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
model tiny {
    vocab_size: 257 max_seq_len: 8 hidden_size: 8 num_layers: 1
    block: {
        attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
        ffn: base
        dropout: 0.0
    }
}
"#;

const MEMORY_MODEL: &str = r#"
ffn base { hidden_dim: 12 activation: swiglu dropout: 0.0 }
memory cms {
    tier fast {
        ffn: base
        reserve_experts { capacity: 1 rank: 3 top_k: 1 }
    }
    tier slow {
        ffn: base residual_init: zero
        reserve_experts { capacity: 2 rank: 3 top_k: 1 }
    }
}
model sleeper {
    vocab_size: 257 max_seq_len: 8 hidden_size: 8 num_layers: 1
    block: {
        attention: { num_heads: 1 dropout: 0.0 position_encoding: none }
        memory: cms
        dropout: 0.0
    }
}
"#;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_hermes-train")
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn run_train(
    model: &Path,
    tokenizer: &Path,
    workflow: &Path,
    output: &Path,
    extra: &[&str],
) -> Output {
    Command::new(binary())
        .arg("train")
        .arg("--config")
        .arg(model)
        .arg("--tokenizer")
        .arg(tokenizer)
        .arg("--workflow")
        .arg(workflow)
        .arg("--output")
        .arg(output)
        .arg("--warmup-steps")
        .arg("0")
        .arg("--checkpoint-every")
        .arg("0")
        .args(extra)
        .output()
        .expect("launching hermes-train")
}

fn diagnostic(output: &Output) -> String {
    format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn signature_stdout(output: &Output) -> String {
    assert!(output.status.success(), "{}", diagnostic(output));
    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert!(stdout.ends_with('\n'), "signature has no trailing newline");
    assert_eq!(
        stdout.lines().count(),
        1,
        "signature output is not one line"
    );
    let signature = stdout.trim_end_matches('\n');
    let digest = signature
        .strip_prefix("sha256:")
        .expect("signature has no sha256 prefix");
    assert_eq!(digest.len(), 64);
    assert!(
        digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "signature is not canonical lowercase SHA-256"
    );
    signature.to_owned()
}

/// Construct the smallest self-contained byte-level BPE accepted by the
/// production tokenizer. IDs 0..=255 are the raw byte alphabet and 256 is EOS.
fn write_tokenizer(path: &Path) {
    let allowed = (33_u8..=126)
        .chain(161..=172)
        .chain(174..=255)
        .collect::<Vec<_>>();
    let mut byte_to_unicode = ['\0'; 256];
    for byte in allowed {
        byte_to_unicode[usize::from(byte)] = char::from(byte);
    }
    let mut offset = 0_u32;
    for byte in 0..=255_u8 {
        if byte_to_unicode[usize::from(byte)] == '\0' {
            byte_to_unicode[usize::from(byte)] = char::from_u32(256 + offset).unwrap();
            offset += 1;
        }
    }
    let vocab = byte_to_unicode
        .iter()
        .enumerate()
        .map(|(byte, character)| (character.to_string(), json!(byte)))
        .collect::<serde_json::Map<_, _>>();
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": [{
            "id": 256,
            "content": "<eos>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": Value::Object(vocab),
            "merges": []
        }
    });
    fs::write(path, serde_json::to_vec_pretty(&tokenizer).unwrap()).unwrap();
}

fn write_common_inputs(root: &Path, model_source: &str) -> (PathBuf, PathBuf, PathBuf) {
    let model = root.join("model.mal");
    let tokenizer = root.join("tokenizer.json");
    let data = root.join("data.jsonl");
    fs::write(&model, model_source).unwrap();
    write_tokenizer(&tokenizer);
    fs::write(&data, b"{\"tokens\":[1,2,3,4,5,6,7,8,9,10,11,12]}\n").unwrap();
    (model, tokenizer, data)
}

fn sleep_json() -> Value {
    let digest = format!("sha256:{}", "0".repeat(64));
    json!({
        "schedule": {
            "clock": "optimizer_steps",
            "terminal_consolidation": "distill_into_base_v1",
            "tiers": [
                {"id": "fast", "update_period": 1, "reserve_slots": 1},
                {"id": "slow", "update_period": 2, "reserve_slots": 2}
            ]
        },
        "knowledge_seeding": {
            "chunk_tokens": 2,
            "teacher_rollouts": 1,
            "detached_student_rollouts": 1,
            "temperature": 1.0,
            "forward_kl_weight": 1.0
        },
        "imitation": {
            "semantic_judge_hash": digest,
            "semantic_weight": 0.5,
            "maximum_edit_distance": 4,
            "grpo_group_size": 2
        },
        "retention_suite": "retention.json",
        "retention_suite_sha256": digest,
        "retention": {
            "evaluator_hash": digest,
            "suite_hash": digest,
            "max_anchor_forward_kl": 1.0,
            "max_anchor_regression": 1.0,
            "min_incorporation_gain": -1.0
        },
        "receiver_learning_rate": 0.0001,
        "receiver_weight_decay": 0.0,
        "grpo_clip_epsilon": 0.2,
        "grpo_advantage_epsilon": 0.000001,
        "grpo_kl_coefficient": 0.0,
        "candidate_directory": "sleep-candidates"
    })
}

fn wake_workflow(data: &Path, periodic_sleep: Option<Value>) -> Value {
    let mut phase = json!({
        "name": "wake",
        "type": "continued_pretrain",
        "task": {"type": "causal_lm"},
        "data": data,
        "sequence_length": 4,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "epochs": 1,
        "shuffle_buffer": 1,
        "steps": 1
    });
    if let Some(sleep) = periodic_sleep {
        phase["periodic_sleep"] = sleep;
    }
    json!({"version": 2, "phases": [phase]})
}

#[test]
fn train_signature_is_read_only_exact_and_needs_no_sleep_runtime() {
    let temporary = TempDir::new().unwrap();
    let (model, tokenizer, data) = write_common_inputs(temporary.path(), MEMORY_MODEL);
    let workflow = temporary.path().join("periodic.json");
    fs::write(
        &workflow,
        serde_json::to_vec_pretty(&wake_workflow(&data, Some(sleep_json()))).unwrap(),
    )
    .unwrap();
    let first_output = temporary.path().join("must-not-exist-a");
    let first = run_train(
        &model,
        &tokenizer,
        &workflow,
        &first_output,
        &["--print-run-signature"],
    );
    let first_signature = signature_stdout(&first);
    assert!(!first_output.exists());
    assert!(!temporary.path().join("sleep-candidates").exists());

    // Output location is not part of training semantics, while the learning
    // rate is. Both calls still stop before creating their requested output.
    let second_output = temporary.path().join("must-not-exist-b");
    let second = run_train(
        &model,
        &tokenizer,
        &workflow,
        &second_output,
        &["--print-run-signature"],
    );
    assert_eq!(signature_stdout(&second), first_signature);
    assert!(!second_output.exists());

    let changed_output = temporary.path().join("must-not-exist-c");
    let changed = run_train(
        &model,
        &tokenizer,
        &workflow,
        &changed_output,
        &["--print-run-signature", "--lr", "0.0004"],
    );
    assert_ne!(signature_stdout(&changed), first_signature);
    assert!(!changed_output.exists());

    let telemetry_output = temporary.path().join("must-not-exist-telemetry");
    let telemetry = run_train(
        &model,
        &tokenizer,
        &workflow,
        &telemetry_output,
        &[
            "--print-run-signature",
            "--gpu-metrics-interval-ms",
            "2500",
            "--gpu-physical-device",
            "GPU-1234-abcd",
        ],
    );
    assert_ne!(signature_stdout(&telemetry), first_signature);
    assert!(!telemetry_output.exists());

    let initial_checkpoint = temporary.path().join("initial.safetensors");
    fs::write(&initial_checkpoint, b"first checkpoint identity").unwrap();
    let checkpoint_output = temporary.path().join("must-not-exist-d");
    let checkpoint_signature = signature_stdout(&run_train(
        &model,
        &tokenizer,
        &workflow,
        &checkpoint_output,
        &[
            "--print-run-signature",
            "--checkpoint",
            initial_checkpoint.to_str().unwrap(),
        ],
    ));
    assert_ne!(checkpoint_signature, first_signature);
    assert!(!checkpoint_output.exists());

    fs::write(&initial_checkpoint, b"second checkpoint identity").unwrap();
    let changed_checkpoint_output = temporary.path().join("must-not-exist-e");
    let changed_checkpoint_signature = signature_stdout(&run_train(
        &model,
        &tokenizer,
        &workflow,
        &changed_checkpoint_output,
        &[
            "--print-run-signature",
            "--checkpoint",
            initial_checkpoint.to_str().unwrap(),
        ],
    ));
    assert_ne!(changed_checkpoint_signature, checkpoint_signature);
    assert!(!changed_checkpoint_output.exists());
}

#[test]
fn workflow_signature_only_matches_runtime_and_mutates_nothing() {
    let temporary = TempDir::new().unwrap();
    let (_, _, data) = write_common_inputs(temporary.path(), ORDINARY_MODEL);
    let workflow_path = temporary.path().join("workflow.json");
    fs::write(
        &workflow_path,
        serde_json::to_vec_pretty(&wake_workflow(&data, None)).unwrap(),
    )
    .unwrap();
    let expected = workflow_signature(&load_workflow(&workflow_path).unwrap()).unwrap();
    let before = fs::read_dir(temporary.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();

    let output = Command::new(binary())
        .arg("validate-workflow")
        .arg("--workflow")
        .arg(&workflow_path)
        .arg("--signature-only")
        .current_dir(temporary.path())
        .output()
        .expect("launching validate-workflow --signature-only");
    assert_eq!(signature_stdout(&output), expected);
    let after = fs::read_dir(temporary.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(after, before);
}

#[test]
fn signature_flags_are_documented_in_command_help() {
    let train_help = Command::new(binary())
        .args(["train", "--help"])
        .output()
        .expect("launching train --help");
    assert!(train_help.status.success(), "{}", diagnostic(&train_help));
    let train_help = String::from_utf8_lossy(&train_help.stdout);
    assert!(train_help.contains("--print-run-signature"));
    assert!(train_help.contains("--gpu-metrics-interval-ms"));
    assert!(train_help.contains("--gpu-physical-device"));

    let workflow_help = Command::new(binary())
        .args(["validate-workflow", "--help"])
        .output()
        .expect("launching validate-workflow --help");
    assert!(
        workflow_help.status.success(),
        "{}",
        diagnostic(&workflow_help)
    );
    assert!(String::from_utf8_lossy(&workflow_help.stdout).contains("--signature-only"));
    assert!(String::from_utf8_lossy(&workflow_help.stdout).contains("--config"));
}

#[test]
fn model_aware_workflow_validation_rejects_sleep_reserve_drift_without_state() {
    let temporary = TempDir::new().unwrap();
    let (model, _, data) = write_common_inputs(temporary.path(), MEMORY_MODEL);
    let workflow_path = temporary.path().join("workflow.json");
    fs::write(
        &workflow_path,
        serde_json::to_vec_pretty(&wake_workflow(&data, Some(sleep_json()))).unwrap(),
    )
    .unwrap();
    let output = Command::new(binary())
        .arg("validate-workflow")
        .arg("--workflow")
        .arg(&workflow_path)
        .arg("--config")
        .arg(&model)
        .output()
        .expect("launching model-aware workflow validation");
    assert!(output.status.success(), "{}", diagnostic(&output));

    let mut mismatched = wake_workflow(&data, Some(sleep_json()));
    mismatched["phases"][0]["periodic_sleep"]["schedule"]["tiers"][1]["reserve_slots"] = json!(3);
    fs::write(
        &workflow_path,
        serde_json::to_vec_pretty(&mismatched).unwrap(),
    )
    .unwrap();
    let output = Command::new(binary())
        .arg("validate-workflow")
        .arg("--workflow")
        .arg(&workflow_path)
        .arg("--config")
        .arg(&model)
        .output()
        .expect("launching mismatched model-aware workflow validation");
    assert!(!output.status.success(), "{}", diagnostic(&output));
    assert!(diagnostic(&output).contains("preallocates"));
    assert!(!temporary.path().join("runtime.json").exists());
}

#[test]
fn cli_enforces_memory_and_periodic_runtime_pairing() {
    let temporary = TempDir::new().unwrap();
    let (memory_model, tokenizer, data) = write_common_inputs(temporary.path(), MEMORY_MODEL);
    let workflow = temporary.path().join("periodic.json");
    fs::write(
        &workflow,
        serde_json::to_vec_pretty(&wake_workflow(&data, Some(sleep_json()))).unwrap(),
    )
    .unwrap();

    let missing_runtime = run_train(
        &memory_model,
        &tokenizer,
        &workflow,
        &temporary.path().join("memory-output"),
        &[],
    );
    assert!(
        !missing_runtime.status.success(),
        "{}",
        diagnostic(&missing_runtime)
    );
    assert!(
        diagnostic(&missing_runtime).contains(
            "a workflow with periodic_sleep requires --sleep-runtime and --sleep-runtime-sha256"
        ),
        "{}",
        diagnostic(&missing_runtime)
    );

    let ordinary_model = temporary.path().join("ordinary.mal");
    fs::write(&ordinary_model, ORDINARY_MODEL).unwrap();
    let ordinary_periodic = run_train(
        &ordinary_model,
        &tokenizer,
        &workflow,
        &temporary.path().join("ordinary-output"),
        &[],
    );
    assert!(
        !ordinary_periodic.status.success(),
        "{}",
        diagnostic(&ordinary_periodic)
    );
    assert!(
        diagnostic(&ordinary_periodic)
            .contains("periodic_sleep requires a MAL model with an explicit memory hierarchy"),
        "{}",
        diagnostic(&ordinary_periodic)
    );
}

#[test]
fn qat_cli_rejects_a_phase_that_never_reaches_its_target_format() {
    let temporary = TempDir::new().unwrap();
    let (model, tokenizer, data) = write_common_inputs(temporary.path(), ORDINARY_MODEL);
    let workflow = temporary.path().join("inactive-qat.json");
    fs::write(
        &workflow,
        serde_json::to_vec_pretty(&json!({
            "version": 2,
            "phases": [{
                "name": "inactive-binary-qat",
                "type": "quantization",
                "task": {"type": "causal_lm"},
                "data": data,
                "sequence_length": 4,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "steps": 2,
                "quantization": {
                    "format": "binary_g128",
                    "group_size": 128,
                    "start_step": 0,
                    "training": {
                        "type": "qat",
                        "warmup_steps": 2,
                        "straight_through": true
                    }
                }
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let result = run_train(
        &model,
        &tokenizer,
        &workflow,
        &temporary.path().join("inactive-output"),
        &[],
    );
    assert!(!result.status.success(), "{}", diagnostic(&result));
    let diagnostic = diagnostic(&result);
    assert!(
        diagnostic.contains("never trains its target format")
            && diagnostic.contains("contains no target-format"),
        "{diagnostic}"
    );
}

#[test]
fn qat_cli_publishes_a_sealed_candidate_and_resume_authenticates_it() {
    let temporary = TempDir::new().unwrap();
    let (model, tokenizer, data) = write_common_inputs(temporary.path(), ORDINARY_MODEL);
    let workflow = temporary.path().join("qat.json");
    let output = temporary.path().join("output");
    fs::write(
        &workflow,
        serde_json::to_vec_pretty(&json!({
            "version": 2,
            "phases": [{
                "name": "binary-qat",
                "type": "quantization",
                "task": {"type": "causal_lm"},
                "data": data,
                "sequence_length": 4,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "epochs": 1,
                "shuffle_buffer": 1,
                "steps": 1,
                "quantization": {
                    "format": "binary_g128",
                    "group_size": 128,
                    "start_step": 0,
                    "embeddings": true,
                    "lm_head": true,
                    "training": {
                        "type": "qat",
                        "warmup_steps": 0,
                        "straight_through": true
                    }
                }
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    // Keep this low-level assertion next to the lifecycle run: the wake
    // trainer sends every hidden matrix to Muon, so a fake-quantized clone
    // which merely produces *some* gradients is not sufficient.
    let config = parse_mal(ORDINARY_MODEL).unwrap();
    let device = Device::ndarray().autodiff();
    let master = Transformer::new(&config, &device).unwrap();
    let muon_ids = master.muon_parameter_ids();
    let (staged, _) =
        fake_quantized_transformer(&master, UltraQuantFormat::BinaryG128, true, true).unwrap();
    let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
    let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
    let mut gradients = staged.forward_loss(input, target).backward();
    let gradients = GradientsParams::from_module(&mut gradients, &staged);
    let missing = muon_ids
        .iter()
        .filter(|id| gradients.get::<2>(**id).is_none())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "QAT omitted Muon gradients: {missing:?}"
    );

    let first = run_train(&model, &tokenizer, &workflow, &output, &[]);
    assert!(first.status.success(), "{}", diagnostic(&first));

    let candidate_root = fs::read_dir(output.join("quantized-candidates"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir() && !path.file_name().unwrap().to_string_lossy().starts_with('.'))
        .expect("one published QAT candidate");
    let candidate = open_qat_candidate(&candidate_root).unwrap();
    assert!(candidate.archive_manifest_path.is_file());
    assert!(candidate.metrics.quantized_tensors > 0);
    assert!(candidate.metrics.quantized_elements > 0);
    assert!(candidate.metrics.packed_bytes < candidate.metrics.archive_weight_bytes);
    let export = fs::read_to_string(output.join("metrics.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<MetricRecord>(line).unwrap())
        .find_map(|record| match record.event {
            MetricEvent::Quantization(metric) if metric.stage == QuantizationStage::Export => {
                Some(metric)
            }
            _ => None,
        })
        .expect("QAT export metric");
    assert_eq!(export.packed_bytes, Some(candidate.metrics.packed_bytes));
    let sealed_before = fs::read(&candidate.candidate_manifest_path).unwrap();

    let pointer: Value =
        serde_json::from_slice(&fs::read(output.join("current.json")).unwrap()).unwrap();
    let generation = pointer["generation"].as_str().unwrap();
    let state_path = output
        .join("generations")
        .join(generation)
        .join("training-state.json");
    let state: Value = serde_json::from_slice(&fs::read(state_path).unwrap()).unwrap();
    assert_eq!(
        state["quantization"]["manifest"].as_str(),
        Some(candidate.candidate_manifest_path.to_str().unwrap())
    );
    assert!(
        state["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|artifact| {
                artifact["kind"] == "hquant_candidate"
                    && artifact["hash"] == candidate.candidate_manifest_sha256
            })
    );

    // A completed run cannot silently replay its last QAT step. Resume loads
    // checkpoint-v2, authenticates the sealed candidate referenced by it, and
    // finishes idempotently. The write-once candidate remains byte-identical.
    let resumed = run_train(&model, &tokenizer, &workflow, &output, &["--resume"]);
    assert!(resumed.status.success(), "{}", diagnostic(&resumed));
    assert_eq!(
        fs::read(&candidate.candidate_manifest_path).unwrap(),
        sealed_before
    );
    assert_eq!(
        open_qat_candidate(&candidate_root)
            .unwrap()
            .candidate_manifest_sha256,
        candidate.candidate_manifest_sha256
    );
}

fn memory_schedule() -> SleepSchedule {
    SleepSchedule {
        clock: UpdateClock::OptimizerSteps,
        terminal_consolidation: TerminalConsolidation::DistillIntoBaseV1,
        tiers: vec![
            MemoryTierSchedule {
                id: "fast".into(),
                update_period: 1,
                reserve_slots: 1,
            },
            MemoryTierSchedule {
                id: "slow".into(),
                update_period: 2,
                reserve_slots: 2,
            },
        ],
    }
}

fn native_sleep_config(root: &Path, retention_sha256: String) -> InModelSleepConfig {
    InModelSleepConfig {
        schedule: memory_schedule(),
        standalone_trigger_clock: None,
        knowledge_seeding: KnowledgeSeedingConfig {
            chunk_tokens: 2,
            teacher_rollouts: 1,
            detached_student_rollouts: 1,
            temperature: 1.0,
            forward_kl_weight: 1.0,
        },
        imitation: ImitationConfig {
            semantic_judge_hash: format!("sha256:{}", "1".repeat(64)),
            semantic_weight: 0.5,
            maximum_edit_distance: 4,
            grpo_group_size: 2,
        },
        dreaming: None,
        retention_suite: root.join("retention.json"),
        retention_suite_sha256: retention_sha256.clone(),
        retention: RetentionGateConfig {
            evaluator_hash: format!("sha256:{}", "2".repeat(64)),
            suite_hash: retention_sha256,
            max_anchor_forward_kl: 1.0,
            max_anchor_regression: 1.0,
            min_incorporation_gain: -1.0,
        },
        receiver_learning_rate: 1e-4,
        receiver_weight_decay: 0.0,
        grpo_clip_epsilon: 0.2,
        grpo_advantage_epsilon: 1e-6,
        grpo_kl_coefficient: 0.0,
        candidate_directory: root.join("candidates"),
    }
}

#[test]
fn periodic_boundary_partitions_and_persists_exact_tier_state() {
    let temporary = TempDir::new().unwrap();
    let config = parse_mal(MEMORY_MODEL).unwrap();
    let device = Device::ndarray().autodiff();
    device.seed(7);
    let mut model = Transformer::new(&config, &device).unwrap();
    model.activate_memory_slot_all_layers(0, 0).unwrap();

    let schedule = memory_schedule();
    let bank = TierOptimizerBank::new(&model, &schedule, TierOptimizerConfig::default()).unwrap();
    let input = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
    let target = Tensor::<2, Int>::from_data([[2, 3, 4, 5]], &device);
    let mut gradients = model.forward_loss(input, target).backward();
    let (wake, report) = bank
        .partition_and_accumulate(&model, &mut gradients, 1)
        .unwrap();
    assert!(!wake.is_empty());
    assert_eq!(report.accumulated_micro_steps, vec![1, 1]);
    assert!(report.tier_gradient_tensors.iter().all(|count| *count > 0));

    let snapshot = bank.snapshot_bytes().unwrap();
    assert_eq!(snapshot, bank.snapshot_bytes().unwrap());
    let restored =
        TierOptimizerBank::new(&model, &schedule, TierOptimizerConfig::default()).unwrap();
    restored.restore_bytes(&snapshot).unwrap();
    assert_eq!(restored.snapshot_bytes().unwrap(), snapshot);

    let publisher = DurableTierOptimizerPublisher::new(
        restored.clone(),
        temporary.path().join("tier-optimizers"),
        TensorTransactionStore::new(temporary.path().join("tensor-transactions")),
        config,
        device,
    )
    .unwrap();
    let durable_scopes = publisher.publish_checkpoint_scopes().unwrap();
    assert!(
        durable_scopes
            .tiers
            .iter()
            .all(|tier| tier.artifact.is_some() && tier.accumulated_micro_steps == 1)
    );
    let durable_snapshot = restored.snapshot_bytes().unwrap();
    assert_eq!(restored.snapshot_bytes().unwrap(), durable_snapshot);

    let retention = b"{\"examples\":[]}";
    fs::write(temporary.path().join("retention.json"), retention).unwrap();
    let sleep_config = native_sleep_config(temporary.path(), sha256(retention));
    let checkpoint_ref = NativeCheckpointRef::new(
        temporary.path().join("wake.safetensors").to_string_lossy(),
        format!("sha256:{}", "3".repeat(64)),
    )
    .unwrap();
    let mut cursor = NativeSleepCheckpoint::new(
        format!("sha256:{}", "4".repeat(64)),
        "wake",
        checkpoint_ref,
        &model,
        &sleep_config,
        1,
    )
    .unwrap();
    cursor.advance_clock(&model, &sleep_config, 1).unwrap();
    assert_eq!(cursor.sleep.next_due_sender(), Some(0));
    cursor.optimizer_scopes = durable_scopes;
    cursor.validate(&model, &sleep_config).unwrap();

    let encoded = serde_json::to_vec(&cursor).unwrap();
    let resumed: NativeSleepCheckpoint = serde_json::from_slice(&encoded).unwrap();
    resumed.validate(&model, &sleep_config).unwrap();
    assert_eq!(serde_json::to_vec(&resumed).unwrap(), encoded);

    let resumed_bank =
        TierOptimizerBank::new(&model, &schedule, TierOptimizerConfig::default()).unwrap();
    resumed_bank.restore_bytes(&durable_snapshot).unwrap();
    assert_eq!(resumed_bank.snapshot_bytes().unwrap(), durable_snapshot);
}
