//! Safetensors checkpoints shared by training and inference.

use std::path::Path;

use anyhow::{Context, Result};
use burn_store::{ApplyResult, ModuleSnapshot, SafetensorsStore};

use super::{ModelBackend, Transformer};

/// Load strictly: missing, unexpected, and shape-mismatched tensors are errors.
pub fn load_safetensors<B: ModelBackend>(
    model: &mut Transformer<B>,
    path: impl AsRef<Path>,
) -> Result<ApplyResult> {
    let path = path.as_ref();
    let mut store = SafetensorsStore::from_file(path).skip_enum_variants(true);
    model
        .load_from(&mut store)
        .with_context(|| format!("failed to load weights from {}", path.display()))
}

pub fn save_safetensors<B: ModelBackend>(
    model: &Transformer<B>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    let mut store = SafetensorsStore::from_file(path).skip_enum_variants(true);
    model
        .save_into(&mut store)
        .with_context(|| format!("failed to save weights to {}", path.display()))
}
