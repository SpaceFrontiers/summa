//! Safetensors checkpoints shared by training and inference.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, ensure};
use burn_store::{ApplyResult, ModuleSnapshot, SafetensorsStore};

use crate::mal::{MemoryTierInit, ModelDef};

use super::Transformer;

static ATOMIC_SAVE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Load strictly: missing, unexpected, and shape-mismatched tensors are errors.
pub fn load_safetensors(model: &mut Transformer, path: impl AsRef<Path>) -> Result<ApplyResult> {
    let path = path.as_ref();
    load_safetensors_store(
        model,
        SafetensorsStore::from_file(path),
        &format!("checkpoint {}", path.display()),
    )
}

/// Load a strictly validated SafeTensors checkpoint from authenticated bytes.
///
/// This is used by crash-safe resume code after it has read and hashed an
/// immutable checkpoint artifact through an already-authenticated directory
/// handle. Keeping the byte buffer as the store's source prevents a later
/// pathname replacement from changing what Burn applies to the model.
pub fn load_safetensors_bytes(
    model: &mut Transformer,
    bytes: Vec<u8>,
    label: &str,
) -> Result<ApplyResult> {
    load_safetensors_store(model, SafetensorsStore::from_bytes(Some(bytes)), label)
}

fn load_safetensors_store(
    model: &mut Transformer,
    store: SafetensorsStore,
    label: &str,
) -> Result<ApplyResult> {
    let mut store = store.skip_enum_variants(true);
    // `ModuleSnapshot::load_from` may successfully apply the tensors it can
    // match and report the rest through `ApplyResult`.  Load into a private
    // clone so either the complete checkpoint is accepted or the caller's
    // model remains byte-for-byte unchanged.
    let mut loaded = model.clone();
    let result = loaded
        .load_from(&mut store)
        .with_context(|| format!("failed to load weights from {label}"))?;
    ensure!(
        result.errors.is_empty() && result.missing.is_empty() && result.unused.is_empty(),
        "{label} does not match the model topology:\n{result}"
    );
    loaded.validate_memory_checkpoint_state()?;
    loaded.sync_memory_state();
    *model = loaded;
    Ok(result)
}

/// Save a canonical safetensors checkpoint shared by training and inference.
///
/// Canonicalizing the metadata order makes identical model state produce
/// byte-identical files suitable for content-addressed publication.
pub fn save_safetensors(model: &Transformer, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut store = SafetensorsStore::from_file(path).skip_enum_variants(true);
    model
        .save_into(&mut store)
        .with_context(|| format!("failed to save weights to {}", path.display()))?;
    canonicalize_safetensors_header(path)
        .with_context(|| format!("failed to canonicalize weights at {}", path.display()))
}

/// Burn's SafeTensors store supplies metadata through a randomized hash map.
/// Tensor order and bytes are already deterministic, but metadata key order
/// otherwise changes the file digest across identical saves. Rewrite only the
/// JSON header in lexical key order; tensor data stays streamed on disk and is
/// never duplicated in memory.
fn canonicalize_safetensors_header(path: &Path) -> Result<()> {
    const HEADER_LENGTH_BYTES: usize = size_of::<u64>();

    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut encoded_length = [0_u8; HEADER_LENGTH_BYTES];
    file.read_exact(&mut encoded_length)?;
    let header_bytes: usize = u64::from_le_bytes(encoded_length)
        .try_into()
        .context("SafeTensors header does not fit in memory")?;
    ensure!(
        header_bytes <= file_bytes.saturating_sub(HEADER_LENGTH_BYTES as u64) as usize,
        "SafeTensors header exceeds file size"
    );

    let mut original = vec![0_u8; header_bytes];
    file.read_exact(&mut original)?;
    let mut header: serde_json::Value = serde_json::from_slice(&original)?;
    sort_json_objects(&mut header);
    let mut canonical = serde_json::to_vec(&header)?;
    ensure!(
        canonical.len() <= header_bytes,
        "canonical SafeTensors header grew from {header_bytes} to {} bytes",
        canonical.len()
    );
    canonical.resize(header_bytes, b' ');
    file.seek(SeekFrom::Start(HEADER_LENGTH_BYTES as u64))?;
    file.write_all(&canonical)?;
    file.flush()?;
    Ok(())
}

fn sort_json_objects(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                sort_json_objects(value);
            }
        }
        serde_json::Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (_, value) in &mut entries {
                sort_json_objects(value);
            }
            object.extend(entries);
        }
        _ => {}
    }
}

/// Upgrade a conventional checkpoint into an explicitly memory-enabled model.
///
/// Ordinary FFN tensors are remapped into tier zero. All later tiers must use
/// `residual_init: zero`, and every preallocated reserve mask must still be
/// dormant, which makes the upgraded model logit-equivalent to the source.
/// The target topology is never inferred implicitly from checkpoint keys.
pub fn upgrade_safetensors_to_memory(
    target: &mut Transformer,
    source_config: &ModelDef,
    source_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<ApplyResult> {
    validate_upgrade_topology(source_config, target.config())?;
    let source_path = source_path.as_ref();
    let output_path = output_path.as_ref();
    let mut upgraded = target.clone();
    upgraded.prepare_memory_upgrade_state();
    let mut store = SafetensorsStore::from_file(source_path)
        .skip_enum_variants(true)
        .allow_partial(true);
    for layer in 0..source_config.num_layers {
        if target.config().block_for_layer(layer).memory.is_some() {
            store = store.with_key_remapping(
                format!(r"^layers\.{layer}\.feed_forward\."),
                format!("layers.{layer}.memory.tiers.0.feed_forward."),
            );
        }
    }
    let result = upgraded.load_from(&mut store).with_context(|| {
        format!(
            "failed to map conventional checkpoint {} into memory topology",
            source_path.display()
        )
    })?;
    if !result.errors.is_empty() || !result.unused.is_empty() {
        anyhow::bail!("memory checkpoint upgrade did not map cleanly:\n{result}");
    }
    let unexpected_missing = result
        .missing
        .iter()
        .filter(|(path, _)| !is_expected_upgrade_tensor(path))
        .map(|(path, _)| path.as_str())
        .collect::<Vec<_>>();
    if !unexpected_missing.is_empty() {
        anyhow::bail!(
            "memory checkpoint upgrade is missing source tensors outside new memory state: {}",
            unexpected_missing.join(", ")
        );
    }
    upgraded.validate_memory_checkpoint_state()?;
    upgraded.sync_memory_state();
    if upgraded
        .memory_slot_statuses()
        .iter()
        .any(|status| status.active)
    {
        anyhow::bail!("new memory reserve slots must be dormant after upgrade");
    }
    save_safetensors_atomic(&upgraded, output_path)?;
    *target = upgraded;
    Ok(result)
}

/// Publish one complete checkpoint without exposing a partially written output
/// or replacing an existing artifact. The temporary file lives beside the
/// destination, so the final hard-link publication is same-filesystem and
/// atomic. A leftover temporary link is harmless and removed best-effort after
/// the destination name is durable.
fn save_safetensors_atomic(model: &Transformer, output_path: &Path) -> Result<()> {
    let parent = checkpoint_output_parent(output_path);
    let file_name = output_path
        .file_name()
        .context("checkpoint output path has no file name")?
        .to_string_lossy();
    ensure!(
        parent.is_dir(),
        "checkpoint output directory {} does not exist",
        parent.display()
    );
    ensure!(
        !output_path.exists(),
        "refusing to replace existing checkpoint {}",
        output_path.display()
    );

    let sequence = ATOMIC_SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.upgrade-{}-{sequence}.tmp",
        std::process::id()
    ));
    ensure!(
        !temporary.exists(),
        "temporary checkpoint path already exists: {}",
        temporary.display()
    );
    let result = (|| {
        save_safetensors(model, &temporary)?;
        OpenOptions::new().read(true).open(&temporary)?.sync_all()?;
        std::fs::hard_link(&temporary, output_path).with_context(|| {
            format!("atomically publishing checkpoint {}", output_path.display())
        })?;
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result
}

fn checkpoint_output_parent(output_path: &Path) -> &Path {
    output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// An ordinary checkpoint can omit only tensors introduced by the target
/// memory topology. Tier zero's base FFN is *not* new: every one of those
/// tensors must have been remapped from the source checkpoint. Keeping this
/// predicate structural prevents a damaged source checkpoint from being
/// accepted merely because the remapped key happens to contain `.memory.`.
fn is_expected_upgrade_tensor(path: &str) -> bool {
    let mut segments = path.split('.');
    if segments.next() != Some("layers")
        || segments
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .is_none()
        || segments.next() != Some("memory")
        || segments.next() != Some("tiers")
    {
        return false;
    }
    let Some(tier) = segments
        .next()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return false;
    };
    if tier > 0 {
        return segments.next().is_some();
    }
    segments.next() == Some("reserve") && segments.next().is_some()
}

fn validate_upgrade_topology(source: &ModelDef, target: &ModelDef) -> Result<()> {
    for (name, matches) in [
        ("vocab_size", source.vocab_size == target.vocab_size),
        ("max_seq_len", source.max_seq_len == target.max_seq_len),
        ("hidden_size", source.hidden_size == target.hidden_size),
        ("num_layers", source.num_layers == target.num_layers),
        (
            "embeddings",
            serde_json::to_value(&source.embeddings)? == serde_json::to_value(&target.embeddings)?,
        ),
        (
            "output",
            serde_json::to_value(&source.output)? == serde_json::to_value(&target.output)?,
        ),
    ] {
        if !matches {
            anyhow::bail!("memory upgrade requires identical source/target {name}");
        }
    }

    let mut memory_layers = 0;
    for layer in 0..source.num_layers {
        let source_block = source.block_for_layer(layer);
        let target_block = target.block_for_layer(layer);
        if source_block.memory.is_some() {
            anyhow::bail!("source layer {layer} is already memory-enabled");
        }
        for (name, matches) in [
            (
                "attention",
                serde_json::to_value(&source_block.attention)?
                    == serde_json::to_value(&target_block.attention)?,
            ),
            (
                "ssm",
                serde_json::to_value(&source_block.ssm)?
                    == serde_json::to_value(&target_block.ssm)?,
            ),
            (
                "norm",
                serde_json::to_value(&source_block.norm)?
                    == serde_json::to_value(&target_block.norm)?,
            ),
            (
                "norm_position",
                source_block.norm_position == target_block.norm_position,
            ),
            ("residual", source_block.residual == target_block.residual),
            ("dropout", source_block.dropout == target_block.dropout),
        ] {
            if !matches {
                anyhow::bail!("memory upgrade layer {layer} changes {name}");
            }
        }
        match &target_block.memory {
            Some(memory) => {
                memory_layers += 1;
                let fast = memory.tiers.first().ok_or_else(|| {
                    anyhow::anyhow!("target layer {layer} memory has no fast tier")
                })?;
                if serde_json::to_value(&fast.ffn)? != serde_json::to_value(&source_block.ffn)? {
                    anyhow::bail!("target layer {layer} fast tier does not match source FFN");
                }
                if memory
                    .tiers
                    .iter()
                    .skip(1)
                    .any(|tier| tier.residual_init != MemoryTierInit::ResidualZero)
                {
                    anyhow::bail!("target layer {layer} slower tiers must use residual_init: zero");
                }
            }
            None => {
                if serde_json::to_value(&source_block.ffn)?
                    != serde_json::to_value(&target_block.ffn)?
                {
                    anyhow::bail!("target layer {layer} changes its ordinary FFN");
                }
            }
        }
    }
    if memory_layers == 0 {
        anyhow::bail!("target model does not contain a memory chain");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use tempfile::tempdir;

    use super::*;
    use crate::model::Device;

    fn tensor_state(model: &Transformer) -> Vec<(String, burn::tensor::TensorData)> {
        let mut tensors = model
            .collect(None, None, true)
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot.to_data().unwrap()))
            .collect::<Vec<_>>();
        tensors.sort_by(|(left, _), (right, _)| left.cmp(right));
        tensors
    }

    #[test]
    fn safetensors_artifact_is_canonical_across_save_and_reload() {
        let config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: base
                norm: rmsnorm { eps: 1e-5 }
            }
            model canonical {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(53);
        let model = Transformer::new(&config, &device).unwrap();
        let directory = tempdir().unwrap();
        let first = directory.path().join("first.safetensors");
        let second = directory.path().join("second.safetensors");
        let reloaded_path = directory.path().join("reloaded.safetensors");

        save_safetensors(&model, &first).unwrap();
        save_safetensors(&model, &second).unwrap();
        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&second).unwrap()
        );

        device.seed(59);
        let mut reloaded = Transformer::new(&config, &device).unwrap();
        load_safetensors(&mut reloaded, &first).unwrap();
        save_safetensors(&reloaded, &reloaded_path).unwrap();
        assert_eq!(
            std::fs::read(&first).unwrap(),
            std::fs::read(&reloaded_path).unwrap()
        );
    }

    #[test]
    fn strict_load_rejects_partial_checkpoint_without_mutating_destination() {
        let one_layer = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: base
                norm: rmsnorm { eps: 1e-5 }
            }
            model partial {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let two_layers = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: base
                norm: rmsnorm { eps: 1e-5 }
            }
            model destination {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 2 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(61);
        let partial = Transformer::new(&one_layer, &device).unwrap();
        device.seed(67);
        let mut destination = Transformer::new(&two_layers, &device).unwrap();
        let directory = tempdir().unwrap();
        let partial_path = directory.path().join("partial.safetensors");
        let two_layer_path = directory.path().join("two-layers.safetensors");
        save_safetensors(&partial, &partial_path).unwrap();
        save_safetensors(&destination, &two_layer_path).unwrap();

        let state_before = tensor_state(&destination);

        let input = Tensor::<2, Int>::from_data([[1, 3, 5, 7]], &device);
        let logits_before = destination
            .forward(input.clone(), 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();

        let error = load_safetensors(&mut destination, &partial_path).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("Missing Tensors"), "{error}");

        let logits_after = destination
            .forward(input, 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(logits_after, logits_before);
        assert_eq!(tensor_state(&destination), state_before);

        // The inverse mismatch is more subtle: Burn applies every destination
        // tensor and reports the extra source layer only through
        // `ApplyResult::unused`. Strict loading must reject that result too.
        device.seed(71);
        let mut smaller_destination = Transformer::new(&one_layer, &device).unwrap();
        let smaller_before = tensor_state(&smaller_destination);
        let error = load_safetensors(&mut smaller_destination, &two_layer_path)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not match the model topology"),
            "{error}"
        );
        assert_eq!(tensor_state(&smaller_destination), smaller_before);
    }

    #[test]
    fn checkpoint_upgrade_failures_leave_target_and_output_unchanged() {
        let source_config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: base
                norm: rmsnorm { eps: 1e-5 }
            }
            model source {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let target_config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            ffn slow { hidden_dim: 8 activation: swiglu bias: false }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                }
                tier slow {
                    ffn: slow
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                    residual_init: zero
                }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                memory: cms
                norm: rmsnorm { eps: 1e-5 }
            }
            model target {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(73);
        let source = Transformer::new(&source_config, &device).unwrap();
        device.seed(79);
        let mut target = Transformer::new(&target_config, &device).unwrap();
        target.activate_memory_slot(0, 0, 0).unwrap();
        let target_before = tensor_state(&target);
        let directory = tempdir().unwrap();
        let source_path = directory.path().join("source.safetensors");
        let invalid_path = directory.path().join("invalid.safetensors");
        let output_path = directory.path().join("upgraded.safetensors");
        save_safetensors(&source, &source_path).unwrap();
        std::fs::write(&invalid_path, b"not-safetensors").unwrap();

        upgrade_safetensors_to_memory(&mut target, &source_config, &invalid_path, &output_path)
            .unwrap_err();
        assert_eq!(tensor_state(&target), target_before);
        assert!(!output_path.exists());

        let sentinel = b"existing-checkpoint-must-not-be-replaced";
        std::fs::write(&output_path, sentinel).unwrap();
        let error =
            upgrade_safetensors_to_memory(&mut target, &source_config, &source_path, &output_path)
                .unwrap_err()
                .to_string();
        assert!(error.contains("refusing to replace"), "{error}");
        assert_eq!(tensor_state(&target), target_before);
        assert_eq!(std::fs::read(&output_path).unwrap(), sentinel);
        assert!(
            std::fs::read_dir(directory.path())
                .unwrap()
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".upgrade-")),
            "failed publication left a temporary checkpoint"
        );
    }

    #[test]
    fn bare_checkpoint_output_uses_current_directory() {
        assert_eq!(
            checkpoint_output_parent(Path::new("upgraded.safetensors")),
            Path::new(".")
        );
    }

    #[test]
    fn conventional_checkpoint_upgrade_preserves_logits() {
        let source_config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: base
                norm: rmsnorm { eps: 1e-5 }
            }
            model source {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let target_config = crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 16 activation: swiglu bias: false }
            ffn slow { hidden_dim: 8 activation: swiglu bias: false }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                }
                tier slow {
                    ffn: slow
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                    residual_init: zero
                }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                memory: cms
                norm: rmsnorm { eps: 1e-5 }
            }
            model target {
                vocab_size: 31 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(71);
        let source = Transformer::new(&source_config, &device).unwrap();
        device.seed(99);
        let mut target = Transformer::new(&target_config, &device).unwrap();
        let directory = tempdir().unwrap();
        let source_path = directory.path().join("source.safetensors");
        let target_path = directory.path().join("target.safetensors");
        let active_target_path = directory.path().join("target-active.safetensors");
        save_safetensors(&source, &source_path).unwrap();
        let upgrade =
            upgrade_safetensors_to_memory(&mut target, &source_config, &source_path, &target_path)
                .unwrap();
        assert!(
            upgrade
                .missing
                .iter()
                .all(|(path, _)| is_expected_upgrade_tensor(path)),
            "upgrade accepted an unexpected missing tensor: {upgrade}"
        );

        let input = Tensor::<2, Int>::from_data([[1, 3, 5, 7]], &device);
        let source_logits = source.forward(input.clone(), 0).into_data();
        let target_logits = target.forward(input, 0).into_data();
        let source_values = source_logits.convert::<f32>().to_vec::<f32>().unwrap();
        let target_values = target_logits.convert::<f32>().to_vec::<f32>().unwrap();
        let maximum = source_values
            .iter()
            .zip(target_values)
            .map(|(source, target)| (source - target).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum < 1e-6, "upgraded logits differ by {maximum}");
        assert!(
            target
                .memory_slot_statuses()
                .iter()
                .all(|slot| !slot.active)
        );

        let dormant_parameters = target.num_params();
        target.activate_memory_slot(0, 1, 0).unwrap();
        let active_parameters = target.num_params();
        assert_eq!(active_parameters, dormant_parameters);
        let active_input = Tensor::<2, Int>::from_data([[2, 4, 6, 8]], &device);
        let active_logits = target.forward(active_input.clone(), 0).into_data();
        save_safetensors(&target, &active_target_path).unwrap();
        let mut reloaded = Transformer::new(&target_config, &device).unwrap();
        load_safetensors(&mut reloaded, &active_target_path).unwrap();
        assert_eq!(reloaded.num_params(), active_parameters);
        assert!(
            tensor_state(&reloaded).iter().all(|(path, _)| {
                !path.contains("routing_mask") && !path.contains("global_to_local")
            }),
            "derived routing caches must not enter checkpoint state"
        );
        let statuses = reloaded.memory_slot_statuses();
        assert!(statuses.iter().any(|slot| slot.tier == 1 && slot.active));
        assert!(statuses.iter().all(|slot| slot.tier == 1 || !slot.active));
        let reloaded_logits = reloaded.forward(active_input, 0).into_data();
        let maximum = active_logits
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(reloaded_logits.convert::<f32>().to_vec::<f32>().unwrap())
            .map(|(active, reloaded)| (active - reloaded).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            maximum < 1e-6,
            "active memory logits changed by {maximum} after roundtrip"
        );
    }

    #[test]
    fn upgrade_missing_tensor_allowlist_is_structural() {
        assert!(is_expected_upgrade_tensor(
            "layers.0.memory.tiers.0.reserve.slots.1.a"
        ));
        assert!(is_expected_upgrade_tensor(
            "layers.17.memory.tiers.2.feed_forward.down_proj.weight"
        ));
        assert!(!is_expected_upgrade_tensor(
            "layers.0.memory.tiers.0.feed_forward.down_proj.weight"
        ));
        assert!(!is_expected_upgrade_tensor(
            "layers.0.memoryish.tiers.1.feed_forward.down_proj.weight"
        ));
        assert!(!is_expected_upgrade_tensor("embedding.weight"));
    }

    #[test]
    fn moe_checkpoint_upgrade_is_reloadable_and_logit_exact() {
        let source_config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12 activation: swiglu bias: false
                moe { experts: 3 top_k: 2 }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                ffn: routed
                norm: rmsnorm { eps: 1e-5 }
            }
            model source {
                vocab_size: 29 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let target_config = crate::mal::parse_mal(
            r#"
            ffn routed {
                hidden_dim: 12 activation: swiglu bias: false
                moe { experts: 3 top_k: 2 }
            }
            ffn slow { hidden_dim: 4 activation: swiglu bias: false }
            memory cms {
                tier fast {
                    ffn: routed
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                }
                tier slow {
                    ffn: slow
                    reserve_experts { capacity: 2 rank: 2 top_k: 1 }
                    residual_init: zero
                }
            }
            block b {
                attention: { num_heads: 2 num_kv_heads: 1 head_dim: 4 position_encoding: none }
                memory: cms
                norm: rmsnorm { eps: 1e-5 }
            }
            model target {
                vocab_size: 29 max_seq_len: 8 hidden_size: 8 num_layers: 1 block: b
                embeddings { tie_weights: true }
            }
            "#,
        )
        .unwrap();
        let device = Device::ndarray();
        device.seed(113);
        let source = Transformer::new(&source_config, &device).unwrap();
        device.seed(127);
        let mut upgraded = Transformer::new(&target_config, &device).unwrap();
        let directory = tempdir().unwrap();
        let source_path = directory.path().join("source-moe.safetensors");
        let target_path = directory.path().join("target-moe.safetensors");
        save_safetensors(&source, &source_path).unwrap();
        upgrade_safetensors_to_memory(&mut upgraded, &source_config, &source_path, &target_path)
            .unwrap();

        let mut reloaded = Transformer::new(&target_config, &device).unwrap();
        load_safetensors(&mut reloaded, &target_path).unwrap();
        let input = Tensor::<2, Int>::from_data([[2, 4, 6, 8]], &device);
        let source_values = source
            .forward(input.clone(), 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let target_values = reloaded
            .forward(input, 0)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let maximum = source_values
            .iter()
            .zip(target_values)
            .map(|(source, target)| (source - target).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum < 1e-6, "reloaded MoE logits differ by {maximum}");
    }
}
