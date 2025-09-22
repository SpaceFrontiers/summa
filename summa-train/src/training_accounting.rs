//! Resource accounting sealed inside training checkpoints.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::artifact_io::validate_sha256_hex;
pub const TRAINING_ACCOUNTING_VERSION: u32 = 1;

pub const TRAINING_ACCOUNTING_FILE: &str = "training-accounting.json";

/// Resource measurements sealed *inside* a checkpoint generation. This layer
/// intentionally does not contain the generation-manifest hash, avoiding a
/// cycle while letting the manifest authenticate every accounting field.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingAccounting {
    pub version: u32,
    pub training_gpu_hours: f64,
    pub parameters: u64,
    pub routed_active_parameters: u64,
    pub weights_bytes: u64,
    pub weights_sha256: String,
}

impl TrainingAccounting {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == TRAINING_ACCOUNTING_VERSION,
            "unsupported training-accounting version {}",
            self.version
        );
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "training accounting has invalid measured GPU hours"
        );
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "training accounting has an empty parameter count"
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "training accounting routes more parameters than it stores"
        );
        ensure!(self.weights_bytes > 0, "training accounting has no weights");
        validate_sha256_hex(&self.weights_sha256, "training-accounting model weights")
            .context("invalid model-weights hash in training accounting")?;
        Ok(())
    }
}
