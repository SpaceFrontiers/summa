//! Structural fairness checks for static-versus-memory benchmark pairs.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use summa_llm::{BlockDef, MemoryTierInit, ModelDef};

/// Verify that the candidate is an additive sleep-memory variant of the
/// reference rather than a differently sized or otherwise easier backbone.
pub fn validate_matched_backbone(static_model: &ModelDef, memory_model: &ModelDef) -> Result<()> {
    ensure!(
        static_model.vocab_size == memory_model.vocab_size
            && static_model.max_seq_len == memory_model.max_seq_len
            && static_model.hidden_size == memory_model.hidden_size
            && static_model.num_layers == memory_model.num_layers,
        "static and memory models must have identical vocabulary, context, hidden size, and layer count"
    );
    ensure!(
        canonical_value(&static_model.embeddings)? == canonical_value(&memory_model.embeddings)?,
        "static and memory embedding configurations differ"
    );
    ensure!(
        canonical_value(&static_model.output)? == canonical_value(&memory_model.output)?,
        "static and memory output configurations differ"
    );

    for layer in 0..static_model.num_layers {
        let static_block = static_model.block_for_layer(layer);
        let memory_block = memory_model.block_for_layer(layer);
        ensure!(
            static_block.memory.is_none(),
            "baseline layer {layer} is not static"
        );
        let memory = memory_block
            .memory
            .as_ref()
            .with_context(|| format!("candidate layer {layer} has no memory hierarchy"))?;
        let fast = memory
            .tiers
            .first()
            .with_context(|| format!("candidate layer {layer} has no memory tiers"))?;
        ensure!(
            canonical_value(&static_block.ffn)? == canonical_value(&fast.ffn)?,
            "layer {layer} static FFN does not match the candidate fast tier"
        );
        ensure!(
            matches!(fast.residual_init, MemoryTierInit::Default),
            "layer {layer} fast memory tier does not preserve the static FFN initialization"
        );
        ensure!(
            memory
                .tiers
                .iter()
                .skip(1)
                .all(|tier| matches!(tier.residual_init, MemoryTierInit::ResidualZero)),
            "layer {layer} has a later memory tier that is not initialized as a residual no-op"
        );
        ensure!(
            canonical_mixer_block(static_block)? == canonical_mixer_block(memory_block)?,
            "layer {layer} sequence mixer, normalization, residual, or dropout differs"
        );
    }
    Ok(())
}

fn canonical_mixer_block(block: &BlockDef) -> Result<Value> {
    let mut value = serde_json::to_value(block)?;
    let object = value
        .as_object_mut()
        .context("serialized block is not an object")?;
    object.remove("name");
    object.remove("ffn");
    object.remove("memory");
    strip_symbolic_names(&mut value);
    Ok(value)
}

fn canonical_value(value: &impl Serialize) -> Result<Value> {
    let mut value = serde_json::to_value(value)?;
    strip_symbolic_names(&mut value);
    Ok(value)
}

fn strip_symbolic_names(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.remove("name");
            for value in object.values_mut() {
                strip_symbolic_names(value);
            }
        }
        Value::Array(array) => {
            for value in array {
                strip_symbolic_names(value);
            }
        }
        _ => {}
    }
}
