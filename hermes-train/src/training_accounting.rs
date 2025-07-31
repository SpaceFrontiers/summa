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

/// Content-addressed attestation emitted after sealing one immutable
/// checkpoint. The accounting itself is already inside that generation; this
/// outer artifact binds its manifest identity without a circular hash.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingEvidence {
    pub version: u32,
    pub checkpoint_manifest_sha256: String,
    pub accounting_sha256: String,
    /// Built-in single-device trainer: sum of committed optimizer-window
    /// elapsed time divided by 3600, including stalls (not GPU-busy time).
    pub training_gpu_hours: f64,
    /// Exact stored parameters in the instantiated model module.
    pub parameters: u64,
    /// All non-expert parameters plus routers/shared experts and ordinary
    /// top-k expert parameter-equivalent.
    pub routed_active_parameters: u64,
    /// Byte length of the model weights artifact only; optimizer/trainer state
    /// is excluded from model-capacity comparisons.
    pub stored_bytes: u64,
    pub weights_sha256: String,
}

impl TrainingEvidence {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported training-evidence version {}",
            self.version
        );
        validate_sha256_hex(
            &self.checkpoint_manifest_sha256,
            "training-evidence checkpoint manifest",
        )
        .context("invalid checkpoint hash in training evidence")?;
        validate_sha256_hex(&self.accounting_sha256, "training-evidence accounting")
            .context("invalid accounting hash in training evidence")?;
        ensure!(
            self.training_gpu_hours.is_finite() && self.training_gpu_hours > 0.0,
            "training evidence has invalid measured GPU hours"
        );
        ensure!(
            self.parameters > 0 && self.routed_active_parameters > 0,
            "training evidence has an empty parameter count"
        );
        ensure!(
            self.routed_active_parameters <= self.parameters,
            "training evidence routes more parameters than it stores"
        );
        ensure!(
            self.stored_bytes > 0,
            "training evidence has no stored bytes"
        );
        validate_sha256_hex(&self.weights_sha256, "training-evidence model weights")
            .context("invalid model-weights hash in training evidence")?;
        Ok(())
    }
}
