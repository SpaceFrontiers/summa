//! Deterministic serialization for stateful Burn optimizers.
//!
//! Burn's [`ModuleOptimizer`] stores per-parameter contexts in a hash map. Its
//! native `save`/`into_bytes` path therefore emits semantically identical
//! tensors in an arbitrary order, which is unsuitable for content-addressed
//! checkpoints. We round-trip the record through `burn-pack` and write tensors
//! in their fully-qualified name order while preserving all scalar and path
//! metadata.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use burn_optim::ModuleOptimizer;
use burn_pack::{Bytes, Reader, Writer};

static OPTIMIZER_STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct OptimizerStagingGuard(PathBuf);

impl Drop for OptimizerStagingGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Return a byte-identical Burn optimizer artifact for identical optimizer
/// state, independent of the optimizer's internal hash-map iteration order.
pub fn canonical_module_optimizer_bytes(optimizer: &ModuleOptimizer) -> Result<Bytes> {
    let bytes = optimizer
        .into_bytes()
        .context("serializing Burn optimizer state")?;
    canonicalize_optimizer_burnpack(bytes)
}

/// Save a [`ModuleOptimizer`] using canonical tensor ordering.
///
/// The caller remains responsible for syncing the file and its parent
/// directory as part of the surrounding atomic checkpoint transaction.
pub fn save_canonical_module_optimizer(
    optimizer: &ModuleOptimizer,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    let parent = path.parent().context("optimizer artifact has no parent")?;
    let file_name = path
        .file_name()
        .context("optimizer artifact has no file name")?
        .to_string_lossy();
    let raw_path = parent.join(format!(
        ".{file_name}.unordered-{}-{}.bpk",
        std::process::id(),
        OPTIMIZER_STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let _guard = OptimizerStagingGuard(raw_path.clone());
    optimizer
        .save(&raw_path)
        .context("writing unordered Burn optimizer staging artifact")?;
    canonical_writer(
        Reader::from_file(&raw_path).context("opening Burn optimizer staging artifact")?,
    )?
    .write_to_file(path)
    .context("writing canonical Burn optimizer artifact")?;
    Ok(())
}

fn canonicalize_optimizer_burnpack(bytes: Bytes) -> Result<Bytes> {
    canonical_writer(Reader::from_bytes(bytes).context("reading Burn optimizer record")?)?
        .into_bytes()
        .context("writing canonical Burn optimizer record")
}

fn canonical_writer(reader: Reader) -> Result<Writer> {
    let metadata = reader.metadata().clone();
    let scalars = reader.scalars().clone();
    let mut tensors = reader
        .into_tensors()
        .context("materializing Burn optimizer record")?;
    tensors.sort_by(|left, right| left.name.cmp(&right.name));

    let mut writer = Writer::new(tensors);
    for (key, value) in scalars {
        writer = writer.with_scalar(&key, value);
    }
    for (key, value) in metadata {
        writer = writer.with_metadata(&key, &value);
    }
    Ok(writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_pack::{DType, Tensor};

    fn tensor(name: &str, value: f32, parameter_id: u64) -> Tensor {
        Tensor::new(
            name.into(),
            DType::F32,
            vec![1],
            Some(parameter_id),
            Bytes::from_bytes_vec(value.to_le_bytes().to_vec()),
        )
    }

    fn pack(tensors: Vec<Tensor>) -> Bytes {
        Writer::new(tensors)
            .with_scalar("step", 7_u64.into())
            .with_metadata("42", "block.ffn.weight")
            .into_bytes()
            .unwrap()
    }

    #[test]
    fn canonical_optimizer_record_ignores_input_tensor_order() {
        let forward = pack(vec![
            tensor("42.exp_avg", 1.0, 42),
            tensor("7.exp_avg", 2.0, 7),
        ]);
        let reverse = pack(vec![
            tensor("7.exp_avg", 2.0, 7),
            tensor("42.exp_avg", 1.0, 42),
        ]);

        let forward = canonicalize_optimizer_burnpack(forward).unwrap();
        let reverse = canonicalize_optimizer_burnpack(reverse).unwrap();
        assert_eq!(&*forward, &*reverse);
        assert_eq!(
            &*forward,
            &*canonicalize_optimizer_burnpack(forward.clone()).unwrap()
        );
    }
}
