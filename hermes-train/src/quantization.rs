//! Ultra-low-bit training artifacts inspired by PrismML's public Bonsai format.
//!
//! The published representation is precise even though the transformation
//! used to obtain Bonsai checkpoints is proprietary: groups of 128 binary or
//! ternary weights share one FP16 scale.  Hermes implements that representation,
//! deterministic L2 codecs, progressive fake quantization, and artifact
//! accounting.  It does not claim to reproduce PrismML's undisclosed training
//! algorithm.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result, bail, ensure};
use burn::module::{Module, ModuleMapper, Param, ParamId};
#[cfg(test)]
use burn::tensor::Device;
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};
use half::{bf16, f16};
use hermes_llm::Transformer;
use safetensors::Dtype;
use safetensors::tensor::{Metadata as SafeTensorMetadata, TensorInfo};
use serde::{Deserialize, Serialize};

use crate::artifact_io::{
    StableIdentity, hash_open_file, sha256_identity, validate_sha256_identity,
};
use crate::workflow::{QuantizationConfig, QuantizationFormat, QuantizationTraining};

const MAGIC: &[u8; 8] = b"HQUANT1\0";
const CODEC_VERSION: u16 = 1;
pub const QUANTIZATION_ARCHIVE_VERSION: u32 = 2;
const ARCHIVE_VERSION: u32 = QUANTIZATION_ARCHIVE_VERSION;
pub const BONSAI_GROUP_SIZE: usize = 128;
const MAX_QUANTIZATION_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_QUANTIZATION_ARCHIVE_MEMBERS: usize = 65_536;
const MAX_QUANTIZATION_ARCHIVE_DIRECTORIES: usize = 4_096;
const MAX_QUANTIZATION_ARCHIVE_DEPTH: usize = 32;
const MAX_QUANTIZATION_MEMBER_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSOR_RANK: usize = 32;
const MAX_TENSOR_NAME_BYTES: usize = 4_096;
const MAX_ARCHIVE_MEMBER_PATH_BYTES: usize = 4_096;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UltraQuantFormat {
    /// One sign bit plus one FP16 scale per group: 1.125 bpw at g128.
    BinaryG128,
    /// Three values in two-bit execution slots plus one FP16 scale per group.
    TernaryG128,
    /// Five base-3 trits per byte for compact checkpoints; kernels may repack
    /// this into two-bit execution slots at load time.
    TernaryEntropyG128,
}

impl UltraQuantFormat {
    fn tag(self) -> u8 {
        match self {
            Self::BinaryG128 => 1,
            Self::TernaryG128 => 2,
            Self::TernaryEntropyG128 => 3,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::BinaryG128),
            2 => Ok(Self::TernaryG128),
            3 => Ok(Self::TernaryEntropyG128),
            _ => bail!("unknown ultra-quant format tag {tag}"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationRecipe {
    pub format: UltraQuantFormat,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    /// First optimizer step that uses a fake-quantized forward pass.
    pub fake_quant_start_step: u64,
    /// Optional ternary warm-up before a binary target.
    #[serde(default)]
    pub ternary_warmup_steps: u64,
    /// Weight of full-precision-teacher output divergence.
    pub distillation_weight: f64,
    /// Matrices are quantized end-to-end; scalar norm and scale tensors remain
    /// floating point, matching the public Bonsai scope.
    #[serde(default = "default_true")]
    pub quantize_embeddings: bool,
    #[serde(default = "default_true")]
    pub quantize_lm_head: bool,
}

/// Fully resolved execution plan for a WorkflowV2 quantization phase.
///
/// `QuantizationRecipe` intentionally describes the on-disk codec and the
/// common fake-quantization knobs. This plan retains the workflow-only
/// scheduling and teacher settings so converting a workflow never silently
/// drops information.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowQuantizationPlan {
    pub recipe: QuantizationRecipe,
    pub start_step: u64,
    pub end_step: Option<u64>,
    pub warmup_format: Option<UltraQuantFormat>,
    pub warmup_steps: u64,
    pub training: WorkflowQuantizationTraining,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkflowQuantizationTraining {
    Qat {
        straight_through: bool,
    },
    Distillation {
        teacher_checkpoint: PathBuf,
        teacher_sha256: String,
        temperature: f64,
        loss_weight: f64,
    },
}

impl WorkflowQuantizationPlan {
    pub fn from_workflow(config: &QuantizationConfig) -> Result<Self> {
        let format = workflow_format(config.format);
        let warmup_format = config.warmup_format.map(workflow_format);
        let start_step =
            u64::try_from(config.start_step).context("quantization start_step exceeds u64")?;
        let end_step = config
            .end_step
            .map(u64::try_from)
            .transpose()
            .context("quantization end_step exceeds u64")?;
        ensure!(
            config.group_size == BONSAI_GROUP_SIZE,
            "workflow quantization group_size must be 128"
        );
        ensure!(
            end_step.is_none_or(|end| end > start_step),
            "workflow quantization end_step must be greater than start_step"
        );

        let (warmup_steps, training, distillation_weight) = match &config.training {
            QuantizationTraining::Qat {
                warmup_steps,
                straight_through,
            } => {
                ensure!(
                    *straight_through,
                    "ultra-low-bit QAT requires straight_through=true"
                );
                (
                    u64::try_from(*warmup_steps)
                        .context("quantization warmup_steps exceeds u64")?,
                    WorkflowQuantizationTraining::Qat {
                        straight_through: true,
                    },
                    0.0,
                )
            }
            QuantizationTraining::Distillation {
                teacher_checkpoint,
                teacher_sha256,
                temperature,
                loss_weight,
            } => {
                ensure!(
                    !teacher_checkpoint.as_os_str().is_empty(),
                    "quantization distillation requires teacher_checkpoint"
                );
                validate_sha256_identity(teacher_sha256, "quantization teacher identity")?;
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "quantization distillation temperature must be finite and positive"
                );
                ensure!(
                    loss_weight.is_finite() && *loss_weight >= 0.0,
                    "quantization distillation loss_weight must be finite and non-negative"
                );
                (
                    0,
                    WorkflowQuantizationTraining::Distillation {
                        teacher_checkpoint: teacher_checkpoint.clone(),
                        teacher_sha256: teacher_sha256.clone(),
                        temperature: *temperature,
                        loss_weight: *loss_weight,
                    },
                    *loss_weight,
                )
            }
        };
        ensure!(
            warmup_steps > 0 || warmup_format.is_none(),
            "warmup_format requires a positive QAT warmup_steps"
        );
        ensure!(
            warmup_format.is_none() || warmup_format != Some(format),
            "warmup_format must differ from the target format"
        );
        if warmup_steps > 0 {
            ensure!(
                matches!(training, WorkflowQuantizationTraining::Qat { .. }),
                "quantization warmup is only supported for QAT"
            );
        }

        // Preserve the compact recipe's historical dense-ternary-to-binary
        // schedule when it is exactly representable. Other warm-up formats
        // remain losslessly represented by `format_at` below.
        let recipe_start = if warmup_steps > 0 && warmup_format.is_none() {
            start_step
                .checked_add(warmup_steps)
                .context("quantization warmup schedule overflows u64")?
        } else {
            start_step
        };
        let ternary_warmup_steps = if format == UltraQuantFormat::BinaryG128
            && warmup_format == Some(UltraQuantFormat::TernaryG128)
        {
            warmup_steps
        } else {
            0
        };
        let plan = Self {
            recipe: QuantizationRecipe {
                format,
                group_size: config.group_size,
                fake_quant_start_step: recipe_start,
                ternary_warmup_steps,
                distillation_weight,
                quantize_embeddings: config.embeddings,
                quantize_lm_head: config.lm_head,
            },
            start_step,
            end_step,
            warmup_format,
            warmup_steps,
            training,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        self.recipe.validate()?;
        ensure!(
            self.end_step.is_none_or(|end| end > self.start_step),
            "quantization end_step must be greater than start_step"
        );
        ensure!(
            self.warmup_steps > 0 || self.warmup_format.is_none(),
            "warmup_format requires positive warmup_steps"
        );
        ensure!(
            self.warmup_format.is_none() || self.warmup_format != Some(self.recipe.format),
            "warmup_format must differ from the target format"
        );
        match &self.training {
            WorkflowQuantizationTraining::Qat { straight_through } => ensure!(
                *straight_through,
                "ultra-low-bit QAT requires straight-through gradients"
            ),
            WorkflowQuantizationTraining::Distillation {
                teacher_checkpoint,
                teacher_sha256,
                temperature,
                loss_weight,
            } => {
                ensure!(
                    !teacher_checkpoint.as_os_str().is_empty(),
                    "quantization teacher checkpoint is empty"
                );
                validate_sha256_identity(teacher_sha256, "quantization teacher identity")?;
                ensure!(
                    temperature.is_finite() && *temperature > 0.0,
                    "quantization teacher temperature must be finite and positive"
                );
                ensure!(
                    loss_weight.is_finite() && *loss_weight >= 0.0,
                    "quantization distillation weight must be finite and non-negative"
                );
                ensure!(
                    (self.recipe.distillation_weight - *loss_weight).abs() <= f64::EPSILON,
                    "quantization recipe and training distillation weights disagree"
                );
            }
        }
        let target_start = self
            .start_step
            .checked_add(self.warmup_steps)
            .context("quantization warmup schedule overflows u64")?;
        ensure!(
            self.end_step.is_none_or(|end| end > target_start),
            "quantization interval ends before the target format becomes active"
        );
        Ok(())
    }

    /// Format used by the forward pass at an absolute optimizer step. `None`
    /// means a full-precision forward (before activation, during an unformatted
    /// warm-up, or after the configured interval).
    pub fn format_at(&self, step: u64) -> Option<UltraQuantFormat> {
        if step < self.start_step || self.end_step.is_some_and(|end| step >= end) {
            return None;
        }
        if step - self.start_step < self.warmup_steps {
            return self.warmup_format;
        }
        Some(self.recipe.format)
    }

    /// Ensure the concrete optimizer-step interval assigned to this workflow
    /// phase executes the target codec at least once. Calibration, full-
    /// precision warm-up, and a different warm-up codec are allowed, but they
    /// cannot consume the whole phase before an archive for `recipe.format` is
    /// published.
    pub fn validate_phase_window(&self, phase_start: u64, phase_steps: u64) -> Result<()> {
        self.validate()?;
        ensure!(phase_steps > 0, "quantization phase has no optimizer steps");
        let phase_end = phase_start
            .checked_add(phase_steps)
            .context("quantization phase optimizer window overflows u64")?;
        let target_start = self
            .start_step
            .checked_add(self.warmup_steps)
            .context("quantization target-format start overflows u64")?;
        let target_end = self.end_step.unwrap_or(u64::MAX);
        ensure!(
            phase_start.max(target_start) < phase_end.min(target_end),
            "quantization phase optimizer window [{phase_start}, {phase_end}) contains no target-format {:?} step from configured interval [{target_start}, {})",
            self.recipe.format,
            self.end_step
                .map_or_else(|| "unbounded".to_owned(), |end| end.to_string())
        );
        Ok(())
    }
}

impl TryFrom<&QuantizationConfig> for WorkflowQuantizationPlan {
    type Error = anyhow::Error;

    fn try_from(config: &QuantizationConfig) -> Result<Self> {
        Self::from_workflow(config)
    }
}

fn workflow_format(format: QuantizationFormat) -> UltraQuantFormat {
    match format {
        QuantizationFormat::BinaryG128 => UltraQuantFormat::BinaryG128,
        QuantizationFormat::TernaryG128 => UltraQuantFormat::TernaryG128,
        QuantizationFormat::TernaryEntropyG128 => UltraQuantFormat::TernaryEntropyG128,
    }
}

fn default_group_size() -> usize {
    BONSAI_GROUP_SIZE
}

fn default_true() -> bool {
    true
}

impl QuantizationRecipe {
    /// Resolve the codec/common training portion of WorkflowV2 quantization.
    /// Use [`WorkflowQuantizationPlan::from_workflow`] when executing training
    /// so end-step, warm-up-format, and teacher settings remain available.
    pub fn from_workflow(config: &QuantizationConfig) -> Result<Self> {
        Ok(WorkflowQuantizationPlan::from_workflow(config)?.recipe)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.group_size == BONSAI_GROUP_SIZE,
            "Bonsai-compatible codecs require group_size 128"
        );
        ensure!(
            self.distillation_weight.is_finite() && self.distillation_weight >= 0.0,
            "quantization distillation_weight must be finite and non-negative"
        );
        ensure!(
            self.format == UltraQuantFormat::BinaryG128 || self.ternary_warmup_steps == 0,
            "ternary_warmup_steps is only meaningful for a binary target"
        );
        self.fake_quant_start_step
            .checked_add(self.ternary_warmup_steps)
            .context("quantization recipe schedule overflows u64")?;
        Ok(())
    }

    pub fn forward_format(&self, step: u64) -> Option<UltraQuantFormat> {
        if step < self.fake_quant_start_step {
            return None;
        }
        if self.format == UltraQuantFormat::BinaryG128
            && step
                < self
                    .fake_quant_start_step
                    .saturating_add(self.ternary_warmup_steps)
        {
            return Some(UltraQuantFormat::TernaryG128);
        }
        Some(self.format)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedTensor {
    pub format: UltraQuantFormat,
    pub shape: Vec<usize>,
    pub group_size: usize,
    /// IEEE FP16 bits, one per group.
    pub scales: Vec<u16>,
    pub codes: Vec<u8>,
}

impl PackedTensor {
    pub fn elements(&self) -> Result<usize> {
        ensure!(
            (1..=MAX_TENSOR_RANK).contains(&self.shape.len()),
            "packed tensor rank must be in 1..={MAX_TENSOR_RANK}"
        );
        checked_elements(&self.shape)
    }

    /// Codec payload rate (codes plus FP16 group scales), excluding the small
    /// HQUANT container header/checksum. Archive accounting uses actual files.
    pub fn bits_per_weight(&self) -> Result<f64> {
        validate_packed(self)?;
        let payload_bytes = self
            .scales
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .and_then(|scale_bytes| scale_bytes.checked_add(self.codes.len()))
            .context("packed tensor payload size overflows usize")?;
        Ok(8.0 * payload_bytes as f64 / self.elements()? as f64)
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        validate_packed(self)?;
        let elements = self.elements()?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(elements)
            .context("reserving decoded quantized tensor")?;
        for group in 0..self.scales.len() {
            let scale = f16::from_bits(self.scales[group]).to_f32();
            match self.format {
                UltraQuantFormat::BinaryG128 => {
                    let start = group * (self.group_size / 8);
                    for index in 0..self.group_size {
                        let bit = (self.codes[start + index / 8] >> (index % 8)) & 1;
                        output.push(if bit == 0 { -scale } else { scale });
                    }
                }
                UltraQuantFormat::TernaryG128 => {
                    let start = group * (self.group_size / 4);
                    for index in 0..self.group_size {
                        let code = (self.codes[start + index / 4] >> ((index % 4) * 2)) & 3;
                        let value = match code {
                            0 => -scale,
                            1 => 0.0,
                            2 => scale,
                            _ => bail!("reserved ternary code 3 in group {group}"),
                        };
                        output.push(value);
                    }
                }
                UltraQuantFormat::TernaryEntropyG128 => {
                    let bytes_per_group = self.group_size.div_ceil(5);
                    let start = group * bytes_per_group;
                    for byte in &self.codes[start..start + bytes_per_group] {
                        ensure!(*byte < 243, "invalid base-3 packed byte {byte}");
                        let mut packed = *byte;
                        for _ in 0..5 {
                            if output.len() == (group + 1) * self.group_size {
                                break;
                            }
                            let trit = packed % 3;
                            packed /= 3;
                            output.push(match trit {
                                0 => -scale,
                                1 => 0.0,
                                _ => scale,
                            });
                        }
                    }
                }
            }
        }
        output.truncate(elements);
        Ok(output)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        validate_packed(self)?;
        let mut payload = Vec::new();
        payload.extend_from_slice(MAGIC);
        payload.extend_from_slice(&CODEC_VERSION.to_le_bytes());
        payload.push(self.format.tag());
        payload.extend_from_slice(&(self.group_size as u16).to_le_bytes());
        payload.push(u8::try_from(self.shape.len()).context("tensor rank exceeds u8")?);
        for &dimension in &self.shape {
            payload.extend_from_slice(&(dimension as u64).to_le_bytes());
        }
        payload.extend_from_slice(&(self.scales.len() as u64).to_le_bytes());
        payload.extend_from_slice(&(self.codes.len() as u64).to_le_bytes());
        for &scale in &self.scales {
            payload.extend_from_slice(&scale.to_le_bytes());
        }
        payload.extend_from_slice(&self.codes);
        let checksum = fnv1a64(&payload);
        payload.extend_from_slice(&checksum.to_le_bytes());
        Ok(payload)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= MAGIC.len() + 2 + 1 + 2 + 1 + 8,
            "quantized tensor is truncated"
        );
        let (payload, checksum_bytes) = bytes.split_at(bytes.len() - 8);
        let expected = u64::from_le_bytes(checksum_bytes.try_into().unwrap());
        ensure!(
            fnv1a64(payload) == expected,
            "quantized tensor checksum mismatch"
        );
        let mut input = Cursor::new(payload);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "invalid quantized tensor magic");
        let version = read_u16(&mut input)?;
        ensure!(
            version == CODEC_VERSION,
            "unsupported quantized tensor version {version}"
        );
        let format = UltraQuantFormat::from_tag(read_u8(&mut input)?)?;
        let group_size = usize::from(read_u16(&mut input)?);
        let rank = usize::from(read_u8(&mut input)?);
        ensure!(
            (1..=MAX_TENSOR_RANK).contains(&rank),
            "packed tensor rank must be in 1..={MAX_TENSOR_RANK}"
        );
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(
                usize::try_from(read_u64(&mut input)?).context("tensor dimension exceeds usize")?,
            );
        }
        let elements = checked_elements(&shape)?;
        ensure!(elements > 0, "packed tensor is empty");
        let scale_count =
            usize::try_from(read_u64(&mut input)?).context("scale count exceeds usize")?;
        let code_count =
            usize::try_from(read_u64(&mut input)?).context("code count exceeds usize")?;
        let remaining = payload
            .len()
            .checked_sub(input.position() as usize)
            .context("quantized tensor header exceeds payload")?;
        let declared = scale_count
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(code_count))
            .context("quantized tensor declared sizes overflow")?;
        ensure!(
            remaining == declared,
            "quantized tensor declared payload size mismatch"
        );
        let mut scales = Vec::with_capacity(scale_count);
        for _ in 0..scale_count {
            scales.push(read_u16(&mut input)?);
        }
        let mut codes = vec![0; code_count];
        input.read_exact(&mut codes)?;
        ensure!(
            input.position() as usize == payload.len(),
            "unexpected quantized tensor payload tail"
        );
        let packed = Self {
            format,
            shape,
            group_size,
            scales,
            codes,
        };
        validate_packed(&packed)?;
        Ok(packed)
    }

    pub fn save_atomic(&self, path: &Path) -> Result<()> {
        if path.exists() {
            ensure!(
                Self::load(path)? == *self,
                "refusing to replace different quantized tensor {}",
                path.display()
            );
            return Ok(());
        }
        let temporary = path.with_extension("hquant.tmp");
        ensure!(
            !temporary.exists(),
            "temporary quantized tensor {} already exists",
            temporary.display()
        );
        write_file_synced(&temporary, &self.to_bytes()?)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("failed to publish {}", path.display()))?;
        fs::File::open(path.parent().unwrap_or_else(|| Path::new(".")))?.sync_all()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            read_regular_file_bounded(path, MAX_QUANTIZATION_MEMBER_BYTES, "packed tensor")?;
        Self::from_bytes(&bytes)
    }

    /// Reference packed matrix-vector implementation used for codec and future
    /// accelerator-kernel parity. It reads codes and group scales directly
    /// without materializing a dequantized matrix; this host implementation is
    /// not presented as a production GEMM kernel.
    pub fn matrix_vector(&self, input: &[f32]) -> Result<Vec<f32>> {
        validate_packed(self)?;
        let elements = self.elements()?;
        ensure!(
            self.shape.len() == 2,
            "packed linear weight must be rank two"
        );
        let rows = self.shape[0];
        let columns = self.shape[1];
        ensure!(input.len() == columns, "packed linear input width mismatch");
        ensure!(
            input.iter().all(|value| value.is_finite()),
            "packed linear input is non-finite"
        );
        let mut output = vec![0.0f32; rows];
        for (row, output) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (column, activation) in input.iter().enumerate() {
                let index = row * columns + column;
                sum += self.value_at(index, elements)? * activation;
            }
            *output = sum;
        }
        Ok(output)
    }

    fn value_at(&self, index: usize, elements: usize) -> Result<f32> {
        ensure!(index < elements, "packed tensor index is out of bounds");
        let group = index / self.group_size;
        let local = index % self.group_size;
        let scale = f16::from_bits(self.scales[group]).to_f32();
        match self.format {
            UltraQuantFormat::BinaryG128 => {
                let start = group * (self.group_size / 8);
                let bit = (self.codes[start + local / 8] >> (local % 8)) & 1;
                Ok(if bit == 0 { -scale } else { scale })
            }
            UltraQuantFormat::TernaryG128 => {
                let start = group * (self.group_size / 4);
                match (self.codes[start + local / 4] >> ((local % 4) * 2)) & 3 {
                    0 => Ok(-scale),
                    1 => Ok(0.0),
                    2 => Ok(scale),
                    _ => bail!("reserved ternary code 3 in group {group}"),
                }
            }
            UltraQuantFormat::TernaryEntropyG128 => {
                let bytes_per_group = self.group_size.div_ceil(5);
                let byte = self.codes[group * bytes_per_group + local / 5];
                ensure!(byte < 243, "invalid base-3 packed byte {byte}");
                let divisor = 3_u16.pow((local % 5) as u32);
                match (u16::from(byte) / divisor) % 3 {
                    0 => Ok(-scale),
                    1 => Ok(0.0),
                    _ => Ok(scale),
                }
            }
        }
    }
}

pub fn quantize_tensor(
    values: &[f32],
    shape: Vec<usize>,
    format: UltraQuantFormat,
    group_size: usize,
) -> Result<PackedTensor> {
    ensure!(group_size == BONSAI_GROUP_SIZE, "only g128 is supported");
    ensure!(
        checked_elements(&shape)? == values.len(),
        "tensor shape does not match data"
    );
    ensure!(!values.is_empty(), "cannot quantize an empty tensor");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "quantized tensor contains non-finite values"
    );
    let groups = values.len().div_ceil(group_size);
    let bytes_per_group = match format {
        UltraQuantFormat::BinaryG128 => group_size / 8,
        UltraQuantFormat::TernaryG128 => group_size / 4,
        UltraQuantFormat::TernaryEntropyG128 => group_size.div_ceil(5),
    };
    let mut scales = Vec::with_capacity(groups);
    let mut codes = vec![0; groups * bytes_per_group];
    for (group_index, group) in values.chunks(group_size).enumerate() {
        let (scale, trits) = match format {
            UltraQuantFormat::BinaryG128 => {
                let scale = (group
                    .iter()
                    .map(|value| f64::from(value.abs()))
                    .sum::<f64>()
                    / group.len() as f64) as f32;
                (scale, Vec::new())
            }
            UltraQuantFormat::TernaryG128 | UltraQuantFormat::TernaryEntropyG128 => {
                optimal_ternary(group)
            }
        };
        let scale = f16::from_f32(scale);
        ensure!(
            scale.to_f32().is_finite() && scale.to_f32() >= 0.0,
            "quantization group {group_index} scale is outside finite FP16 range"
        );
        scales.push(scale.to_bits());
        let start = group_index * bytes_per_group;
        match format {
            UltraQuantFormat::BinaryG128 => {
                for (index, value) in group.iter().enumerate() {
                    if *value >= 0.0 {
                        codes[start + index / 8] |= 1 << (index % 8);
                    }
                }
            }
            UltraQuantFormat::TernaryG128 => {
                for (index, &trit) in trits.iter().enumerate() {
                    codes[start + index / 4] |= trit << ((index % 4) * 2);
                }
            }
            UltraQuantFormat::TernaryEntropyG128 => {
                for (byte_index, chunk) in trits.chunks(5).enumerate() {
                    let mut multiplier = 1u8;
                    let mut packed = 0u8;
                    for &trit in chunk {
                        packed += trit * multiplier;
                        multiplier *= 3;
                    }
                    codes[start + byte_index] = packed;
                }
            }
        }
    }
    let packed = PackedTensor {
        format,
        shape,
        group_size,
        scales,
        codes,
    };
    validate_packed(&packed)?;
    Ok(packed)
}

/// Quantize and immediately reconstruct values with the serialized FP16 group
/// scales. Tensor backends use the same deterministic code/scale estimator but
/// retain scales in the training dtype before applying the STE.
pub fn fake_quantize(values: &[f32], format: UltraQuantFormat) -> Result<Vec<f32>> {
    quantize_tensor(values, vec![values.len()], format, BONSAI_GROUP_SIZE)?.decode()
}

/// Apply the g128 training target entirely on the active tensor device and
/// pass gradients through with a straight-through estimator.  This avoids a
/// device-to-host round trip for every matrix on every microbatch.  Compact
/// base-3 ternary archives and two-bit ternary execution slots share the same
/// reconstructed {-scale, 0, +scale} values, so their QAT forwards are
/// intentionally identical.
///
/// The ternary path minimizes squared error independently for every group by
/// sorting magnitudes, evaluating every non-zero prefix, and scattering the
/// selected ranks back to their original positions.  It therefore implements
/// the same pre-serialization L2 target as [`quantize_tensor`], including
/// partial final groups. HQUANT rounds the resulting scale to FP16 at export.
pub fn fake_quantize_tensor<const D: usize>(
    weights: Tensor<D>,
    format: UltraQuantFormat,
) -> Tensor<D> {
    let shape = weights.dims();
    let elements = shape.iter().product::<usize>();
    assert!(elements > 0, "cannot fake-quantize an empty tensor");
    let groups = elements.div_ceil(BONSAI_GROUP_SIZE);
    let padded_elements = groups * BONSAI_GROUP_SIZE;
    let device = weights.device();
    let detached = weights.clone().detach().reshape([elements]);
    let padded = if padded_elements == elements {
        detached
    } else {
        Tensor::cat(
            vec![
                detached,
                Tensor::zeros([padded_elements - elements], &device),
            ],
            0,
        )
    };
    let grouped = padded.reshape([groups, BONSAI_GROUP_SIZE]);
    let final_group = elements % BONSAI_GROUP_SIZE;
    let valid_counts = Tensor::from_data(
        TensorData::new(
            (0..groups)
                .map(|group| {
                    if group + 1 == groups && final_group != 0 {
                        final_group as f32
                    } else {
                        BONSAI_GROUP_SIZE as f32
                    }
                })
                .collect::<Vec<_>>(),
            [groups, 1],
        ),
        &device,
    );
    let quantized = match format {
        UltraQuantFormat::BinaryG128 => {
            let scale = grouped.clone().abs().sum_dim(1) / valid_counts;
            let signs = grouped.clone().greater_equal_elem(0.0).float() * 2.0 - 1.0;
            signs * scale
        }
        UltraQuantFormat::TernaryG128 | UltraQuantFormat::TernaryEntropyG128 => {
            let magnitudes = grouped.clone().abs();
            let (ranked, original_indices) = magnitudes.sort_descending_with_indices(1);
            let prefix = ranked.cumsum(1);
            let total_square = grouped.clone().square().sum_dim(1);
            let counts = Tensor::from_data(
                TensorData::new(
                    (1..=BONSAI_GROUP_SIZE)
                        .map(|value| value as f32)
                        .collect::<Vec<_>>(),
                    [1, BONSAI_GROUP_SIZE],
                ),
                &device,
            );
            let errors = total_square - prefix.clone().square() / counts.clone();
            let best_rank = errors.argmin(1);
            let scale = prefix.gather(1, best_rank.clone()) / (best_rank.clone() + 1).float();
            let ranks = Tensor::<1, Int>::arange(0..BONSAI_GROUP_SIZE as i64, &device)
                .reshape([1, BONSAI_GROUP_SIZE]);
            let ranked_mask = ranks.lower_equal(best_rank).float();
            let selected = Tensor::zeros([groups, BONSAI_GROUP_SIZE], &device).scatter(
                1,
                original_indices,
                ranked_mask,
                IndexingUpdateOp::Add,
            );
            grouped.sign() * selected * scale
        }
    };
    let reconstructed = quantized
        .reshape([padded_elements])
        .slice(0..elements)
        .reshape(shape);
    // q is detached by construction. Writing the expression in this form
    // makes the forward exactly q and the derivative with respect to the
    // full-precision master exactly one.
    weights.clone() + (reconstructed - weights).detach()
}

struct FakeQuantTransformerMapper<'a> {
    selected: &'a [ParamId],
    format: UltraQuantFormat,
    mapped: usize,
}

impl ModuleMapper for FakeQuantTransformerMapper<'_> {
    fn map_float<const D: usize>(&mut self, parameter: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = parameter.consume();
        let tensor = if D >= 2 && self.selected.contains(&id) {
            self.mapped += 1;
            // `fake_quantize_tensor` expresses the STE against its source
            // leaf. Once that expression is installed as a Param, Burn's
            // optimizer visitor asks the *mapped output* for its gradient;
            // non-leaf outputs do not retain one. The temporary forward model
            // is already an optimizer proxy for the full-precision master, so
            // make the reconstructed value a fresh leaf with the same ParamId.
            // Its direct gradient is exactly the STE gradient applied to the
            // master by the trainer.
            let requires_grad = tensor.is_require_grad();
            fake_quantize_tensor(tensor, self.format)
                .detach()
                .set_require_grad(requires_grad)
        } else {
            tensor
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// Clone a Transformer with selected stored weights replaced by on-device STE
/// fake-quantized values. Parameter IDs are retained, so gradients produced by
/// this forward clone apply directly to the authoritative full-precision
/// master model and its existing optimizer state.
pub fn fake_quantized_transformer(
    master: &Transformer,
    format: UltraQuantFormat,
    quantize_embeddings: bool,
    quantize_lm_head: bool,
) -> Result<(Transformer, usize)> {
    let selected = master
        .ultra_quant_parameter_ids(quantize_embeddings, quantize_lm_head)
        .context("resolve Transformer QAT parameter policy")?;
    let mut mapper = FakeQuantTransformerMapper {
        selected: &selected,
        format,
        mapped: 0,
    };
    let staged = master.clone().map(&mut mapper);
    ensure!(
        mapper.mapped == selected.len(),
        "QAT selected {} parameter tensors but mapped {}",
        selected.len(),
        mapper.mapped
    );
    Ok((staged, mapper.mapped))
}

fn optimal_ternary(group: &[f32]) -> (f32, Vec<u8>) {
    let mut ranked = group
        .iter()
        .enumerate()
        .map(|(index, value)| (index, f64::from(value.abs())))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    let magnitudes = ranked
        .iter()
        .map(|(_, magnitude)| *magnitude)
        .collect::<Vec<_>>();
    let total_square = magnitudes.iter().map(|value| value * value).sum::<f64>();
    let mut prefix = 0.0f64;
    let mut best_error = total_square;
    let mut best_nonzero = 0usize;
    let mut best_scale = 0.0f64;
    for (index, magnitude) in magnitudes.iter().enumerate() {
        prefix += magnitude;
        let nonzero = index + 1;
        let error = total_square - prefix * prefix / nonzero as f64;
        if error < best_error {
            best_error = error;
            best_nonzero = nonzero;
            best_scale = prefix / nonzero as f64;
        }
    }
    let mut selected = vec![false; group.len()];
    for (index, _) in ranked.into_iter().take(best_nonzero) {
        selected[index] = true;
    }
    let trits = group
        .iter()
        .zip(selected)
        .map(|(value, selected)| {
            if !selected || best_scale == 0.0 {
                1 // zero
            } else if *value < 0.0 {
                0 // -scale
            } else {
                2 // +scale
            }
        })
        .collect();
    (best_scale as f32, trits)
}

fn validate_packed(packed: &PackedTensor) -> Result<()> {
    ensure!(
        packed.group_size == BONSAI_GROUP_SIZE,
        "packed tensor is not g128"
    );
    let elements = packed.elements()?;
    ensure!(elements > 0, "packed tensor is empty");
    let groups = elements.div_ceil(packed.group_size);
    ensure!(
        packed.scales.len() == groups,
        "packed tensor scale count mismatch"
    );
    let expected_codes = groups
        * match packed.format {
            UltraQuantFormat::BinaryG128 => packed.group_size / 8,
            UltraQuantFormat::TernaryG128 => packed.group_size / 4,
            UltraQuantFormat::TernaryEntropyG128 => packed.group_size.div_ceil(5),
        };
    ensure!(
        packed.codes.len() == expected_codes,
        "packed tensor code size mismatch"
    );
    ensure!(
        packed.scales.iter().all(|bits| {
            let scale = f16::from_bits(*bits).to_f32();
            scale.is_finite() && scale >= 0.0 && !scale.is_sign_negative()
        }),
        "packed tensor contains an invalid or non-canonical scale"
    );
    match packed.format {
        UltraQuantFormat::TernaryG128 => ensure!(
            packed
                .codes
                .iter()
                .all(|byte| { (0..4).all(|slot| ((byte >> (slot * 2)) & 3) != 3) }),
            "packed tensor contains reserved ternary code 3"
        ),
        UltraQuantFormat::TernaryEntropyG128 => ensure!(
            packed.codes.iter().all(|byte| *byte < 243),
            "packed tensor contains an invalid base-3 byte"
        ),
        UltraQuantFormat::BinaryG128 => {}
    }
    validate_zero_padding(packed, elements)?;
    Ok(())
}

/// HQUANT permits a partial final g128 group, unlike GGML's fixed-size block
/// contract. Keep every unused code bit/trit canonical so equivalent payloads
/// cannot acquire different content identities and future block kernels never
/// observe attacker-controlled padding lanes.
fn validate_zero_padding(packed: &PackedTensor, elements: usize) -> Result<()> {
    let groups = elements.div_ceil(packed.group_size);
    let valid_in_last = elements - (groups - 1) * packed.group_size;
    let slots_per_byte = match packed.format {
        UltraQuantFormat::BinaryG128 => 8,
        UltraQuantFormat::TernaryG128 => 4,
        UltraQuantFormat::TernaryEntropyG128 => 5,
    };
    let bytes_per_group = packed.codes.len() / groups;
    let encoded_slots = bytes_per_group * slots_per_byte;
    let last_group_start = (groups - 1) * bytes_per_group;

    for slot in valid_in_last..encoded_slots {
        let byte = packed.codes[last_group_start + slot / slots_per_byte];
        let code = match packed.format {
            UltraQuantFormat::BinaryG128 => (byte >> (slot % 8)) & 1,
            UltraQuantFormat::TernaryG128 => (byte >> ((slot % 4) * 2)) & 3,
            UltraQuantFormat::TernaryEntropyG128 => {
                let divisor = 3_u16.pow((slot % 5) as u32);
                ((u16::from(byte) / divisor) % 3) as u8
            }
        };
        ensure!(
            code == 0,
            "packed tensor contains non-canonical final-group padding"
        );
    }

    // Base-3 packing has 130 physical slots for every full 128-value group.
    // Its two high padding trits are canonical zero in each preceding group.
    if packed.format == UltraQuantFormat::TernaryEntropyG128 {
        for group in 0..groups.saturating_sub(1) {
            let last_byte = packed.codes[group * bytes_per_group + bytes_per_group - 1];
            ensure!(
                last_byte < 27,
                "packed tensor contains non-canonical base-3 group padding"
            );
        }
    }
    Ok(())
}

fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .context("tensor element count overflows usize")
    })
}

fn checked_sum(mut values: impl Iterator<Item = u64>, label: &str) -> Result<u64> {
    values.try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .with_context(|| format!("{label} overflows u64"))
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizedMatrixManifest {
    pub name: String,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub format: UltraQuantFormat,
    pub file: String,
    pub packed_bytes: u64,
    pub sha256: String,
    pub mean_squared_error: f64,
    pub maximum_absolute_error: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FloatingTensorManifest {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationManifest {
    pub version: u32,
    pub base_checkpoint_hash: String,
    pub recipe: QuantizationRecipe,
    pub matrices: Vec<QuantizedMatrixManifest>,
    pub floating_tensors: Vec<FloatingTensorManifest>,
}

impl QuantizationManifest {
    pub fn true_average_bits_per_weight(&self) -> Result<f64> {
        ensure!(
            self.version == ARCHIVE_VERSION,
            "unsupported quantization manifest version {}",
            self.version
        );
        ensure!(
            !self.base_checkpoint_hash.trim().is_empty(),
            "base checkpoint hash is empty"
        );
        self.recipe.validate()?;
        let quantized_elements = checked_sum(
            self.matrices.iter().map(|matrix| matrix.elements),
            "quantized element count",
        )?;
        let floating_elements = checked_sum(
            self.floating_tensors.iter().map(|tensor| tensor.elements),
            "floating element count",
        )?;
        let total_elements = quantized_elements
            .checked_add(floating_elements)
            .context("archive total element count overflows u64")?;
        ensure!(total_elements > 0, "quantization manifest has no weights");
        let quantized_bytes = checked_sum(
            self.matrices.iter().map(|matrix| matrix.packed_bytes),
            "quantized byte count",
        )?;
        let floating_bytes = checked_sum(
            self.floating_tensors.iter().map(|tensor| tensor.bytes),
            "floating byte count",
        )?;
        let total_bytes = quantized_bytes
            .checked_add(floating_bytes)
            .context("archive total byte count overflows u64")?;
        Ok(total_bytes as f64 * 8.0 / total_elements as f64)
    }
}

/// Validated ultra-low-bit weight archive. `open` verifies every member,
/// not only the manifest, before exposing any tensor to a model backend.
#[derive(Clone, Debug)]
pub struct QuantizedArchive {
    root: PathBuf,
    manifest: QuantizationManifest,
    manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloatingTensorData {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Transactional adapter for installing a validated archive into a serving or
/// training model representation. A failed member load always invokes
/// `rollback_archive_load`; implementations publish changes only in `commit`.
pub trait QuantizedModelBackend {
    fn begin_archive_load(&mut self, manifest: &QuantizationManifest) -> Result<()>;
    fn load_quantized_matrix(&mut self, name: &str, tensor: &PackedTensor) -> Result<()>;
    fn load_floating_tensor(&mut self, tensor: &FloatingTensorData) -> Result<()>;
    fn commit_archive_load(&mut self) -> Result<()>;
    fn rollback_archive_load(&mut self) -> Result<()>;
}

impl QuantizedArchive {
    pub fn open(root: &Path) -> Result<Self> {
        Self::open_inner(root, None)
    }

    /// Open an archive whose exact manifest bytes are sealed by an enclosing
    /// immutable receipt. Member hashes are read only from those authenticated
    /// bytes, so pathname replacement cannot mix one manifest with another
    /// archive generation.
    pub(crate) fn open_addressed(root: &Path, expected_manifest_sha256: &str) -> Result<Self> {
        validate_sha256_identity(expected_manifest_sha256, "quantization manifest identity")
            .context("invalid expected quantization manifest identity")?;
        Self::open_inner(root, Some(expected_manifest_sha256))
    }

    fn open_inner(root: &Path, expected_manifest_sha256: Option<&str>) -> Result<Self> {
        Self::open_inner_with_hook(root, expected_manifest_sha256, || Ok(()))
    }

    fn open_inner_with_hook(
        root: &Path,
        expected_manifest_sha256: Option<&str>,
        after_manifest: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let metadata = fs::symlink_metadata(root)
            .with_context(|| format!("failed to inspect quantized archive {}", root.display()))?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "quantized archive root must be a real directory"
        );
        let manifest_path = root.join("manifest.json");
        let manifest_metadata = fs::symlink_metadata(&manifest_path).with_context(|| {
            format!(
                "failed to inspect quantization manifest {}",
                manifest_path.display()
            )
        })?;
        ensure!(
            manifest_metadata.is_file() && !manifest_metadata.file_type().is_symlink(),
            "quantization manifest must be a regular file"
        );
        let manifest_bytes = read_regular_file_bounded(
            &manifest_path,
            MAX_QUANTIZATION_MANIFEST_BYTES,
            "quantization manifest",
        )?;
        let manifest_sha256 = sha256_identity(&manifest_bytes);
        if let Some(expected) = expected_manifest_sha256 {
            ensure!(
                manifest_sha256 == expected,
                "quantization manifest identity mismatch"
            );
        }
        let manifest: QuantizationManifest =
            serde_json::from_slice(&manifest_bytes).context("invalid quantization manifest")?;
        manifest.true_average_bits_per_weight()?;
        ensure!(
            manifest.version == ARCHIVE_VERSION,
            "unsupported quantization manifest version {}",
            manifest.version
        );
        validate_sha256_identity(
            &manifest.base_checkpoint_hash,
            "quantization source checkpoint identity",
        )
        .context("quantization manifest has an invalid source checkpoint hash")?;
        ensure!(
            !manifest.matrices.is_empty(),
            "quantization archive contains no quantized matrices"
        );
        ensure!(
            manifest
                .matrices
                .len()
                .checked_add(manifest.floating_tensors.len())
                .is_some_and(|members| members <= MAX_QUANTIZATION_ARCHIVE_MEMBERS),
            "quantization archive exceeds its {MAX_QUANTIZATION_ARCHIVE_MEMBERS}-member limit"
        );
        after_manifest()?;

        let mut names = BTreeSet::new();
        let mut files = BTreeSet::new();
        for matrix in &manifest.matrices {
            validate_member_identity(&matrix.name, &matrix.file, &mut names, &mut files)?;
            ensure!(
                (2..=MAX_TENSOR_RANK).contains(&matrix.shape.len()),
                "quantized matrix `{}` rank must be in 2..={MAX_TENSOR_RANK}",
                matrix.name,
            );
            ensure!(
                matrix.elements
                    == u64::try_from(checked_elements(&matrix.shape)?)
                        .context("matrix element count exceeds u64")?,
                "quantized matrix `{}` element count mismatch",
                matrix.name
            );
            ensure!(
                matrix.packed_bytes
                    == packed_tensor_storage_bytes(
                        &matrix.shape,
                        matrix.format,
                        manifest.recipe.group_size,
                    )?,
                "quantized matrix `{}` packed byte count does not match its shape and format",
                matrix.name
            );
            ensure!(
                matrix.mean_squared_error.is_finite() && matrix.mean_squared_error >= 0.0,
                "quantized matrix `{}` has invalid mean-squared error",
                matrix.name
            );
            ensure!(
                matrix.maximum_absolute_error.is_finite() && matrix.maximum_absolute_error >= 0.0,
                "quantized matrix `{}` has invalid maximum error",
                matrix.name
            );
            let path = checked_archive_member(root, &matrix.file)?;
            let bytes = read_verified_member(&path, matrix.packed_bytes, &matrix.sha256)?;
            let packed = PackedTensor::from_bytes(&bytes)
                .with_context(|| format!("invalid quantized matrix `{}`", matrix.name))?;
            ensure!(
                matrix.format == manifest.recipe.format
                    && packed.format == matrix.format
                    && packed.shape == matrix.shape
                    && packed.group_size == manifest.recipe.group_size,
                "quantized matrix `{}` does not match its manifest",
                matrix.name
            );
        }
        for tensor in &manifest.floating_tensors {
            validate_member_identity(&tensor.name, &tensor.file, &mut names, &mut files)?;
            ensure!(
                tensor.shape.len() <= MAX_TENSOR_RANK,
                "floating tensor `{}` has unsupported rank {}",
                tensor.name,
                tensor.shape.len()
            );
            let elements = checked_elements(&tensor.shape)?;
            ensure!(
                tensor.elements
                    == u64::try_from(elements).context("floating element count exceeds u64")?,
                "floating tensor `{}` element count mismatch",
                tensor.name
            );
            let expected_bytes = tensor_storage_bytes(&tensor.dtype, elements)?;
            ensure!(
                tensor.bytes == expected_bytes,
                "floating tensor `{}` byte count does not match dtype and shape",
                tensor.name
            );
            let path = checked_archive_member(root, &tensor.file)?;
            read_verified_member(&path, tensor.bytes, &tensor.sha256)?;
        }
        validate_archive_inventory(root, &files)?;

        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            manifest_sha256,
        })
    }

    pub fn manifest(&self) -> &QuantizationManifest {
        &self.manifest
    }

    pub(crate) fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// Content identity of the parsed manifest. Since every member SHA-256 is
    /// embedded in schema v2, this also identifies the complete archive.
    pub fn content_hash(&self) -> Result<String> {
        Ok(sha256_identity(&serde_json::to_vec(&self.manifest)?))
    }

    /// Authenticate the archive's source identity without regenerating its
    /// quantized tensors.
    ///
    /// `source_sha256` must come from an independently authenticated source
    /// checkpoint (for example, a sealed trainer generation). [`Self::open`]
    /// has already authenticated every archive member against the hashes in
    /// the manifest, so this comparison binds those exact bytes to that exact
    /// source without repeating a potentially multi-gigabyte quantization.
    pub fn verify_source_identity(&self, source_sha256: &str) -> Result<()> {
        validate_sha256_identity(source_sha256, "quantization source checkpoint identity")
            .context("invalid source checkpoint identity")?;
        ensure!(
            self.manifest.base_checkpoint_hash == source_sha256,
            "quantized archive source checkpoint hash mismatch"
        );
        Ok(())
    }

    /// Reproduce the complete archive from a source checkpoint.
    ///
    /// This deliberately expensive derivation check is required once while a
    /// new archive is created. Routine authenticated reopens should use
    /// [`Self::verify_source_identity`] instead: member hashes plus the sealed
    /// source digest already preserve the same immutable binding.
    pub fn verify_source_checkpoint(&self, checkpoint: &Path) -> Result<()> {
        let mut source = StreamingSafetensors::open(checkpoint, "source checkpoint")?;
        self.verify_source_safetensors(&mut source)
    }

    fn verify_source_safetensors(&self, source: &mut StreamingSafetensors) -> Result<()> {
        self.verify_source_identity(source.sha256())?;
        ensure!(
            !source.tensors.is_empty(),
            "source safetensors checkpoint contains no tensors"
        );
        let matrices = self
            .manifest
            .matrices
            .iter()
            .map(|matrix| (matrix.name.as_str(), matrix))
            .collect::<BTreeMap<_, _>>();
        let floating = self
            .manifest
            .floating_tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        let archive_names = matrices
            .keys()
            .chain(floating.keys())
            .map(|name| (*name).to_owned())
            .collect::<BTreeSet<_>>();
        let source_names = source.tensors.keys().cloned().collect::<BTreeSet<_>>();
        ensure!(
            archive_names == source_names,
            "quantized archive tensor inventory differs from source checkpoint: expected {source_names:?}, got {archive_names:?}"
        );

        for (name, tensor) in source.tensor_entries() {
            let elements = checked_elements(&tensor.shape)?;
            let elements_u64 =
                u64::try_from(elements).context("source tensor element count exceeds u64")?;
            let tensor_bytes = source.read_tensor(&tensor)?;
            if should_quantize(&name, &tensor.shape, tensor.dtype, &self.manifest.recipe) {
                let matrix = matrices.get(name.as_str()).with_context(|| {
                    format!("source tensor `{name}` must be a quantized matrix under this recipe")
                })?;
                ensure!(
                    matrix.shape == tensor.shape && matrix.elements == elements_u64,
                    "quantized matrix `{name}` shape or element count differs from source checkpoint"
                );
                let values = tensor_as_f32(tensor.dtype, &tensor_bytes)?;
                ensure!(
                    values.len() == elements,
                    "source tensor `{name}` data length does not match its shape"
                );
                let expected = quantize_tensor(
                    &values,
                    tensor.shape.clone(),
                    self.manifest.recipe.format,
                    self.manifest.recipe.group_size,
                )?;
                let actual = self.load_matrix_entry(matrix)?;
                ensure!(
                    actual == expected,
                    "quantized matrix `{name}` does not match deterministic source encoding"
                );
                let decoded = actual.decode()?;
                let (mean_squared_error, maximum_absolute_error) =
                    quantization_error(&values, &decoded);
                ensure!(
                    matrix.mean_squared_error.to_bits() == mean_squared_error.to_bits(),
                    "quantized matrix `{name}` mean-squared error was not derived from source"
                );
                ensure!(
                    matrix.maximum_absolute_error.to_bits() == maximum_absolute_error.to_bits(),
                    "quantized matrix `{name}` maximum error was not derived from source"
                );
            } else {
                let archived = floating.get(name.as_str()).with_context(|| {
                    format!("source tensor `{name}` must remain floating under this recipe")
                })?;
                let dtype = format!("{:?}", tensor.dtype);
                ensure!(
                    archived.dtype == dtype
                        && archived.shape == tensor.shape
                        && archived.elements == elements_u64,
                    "floating tensor `{name}` metadata differs from source checkpoint"
                );
                ensure!(
                    self.load_floating_entry(archived)?.bytes == tensor_bytes,
                    "floating tensor `{name}` bytes differ from source checkpoint"
                );
            }
        }
        source.ensure_unchanged()
    }

    pub fn load_matrix(&self, name: &str) -> Result<PackedTensor> {
        let matrix = self
            .manifest
            .matrices
            .iter()
            .find(|matrix| matrix.name == name)
            .with_context(|| format!("quantized archive has no matrix `{name}`"))?;
        self.load_matrix_entry(matrix)
    }

    fn load_matrix_entry(&self, matrix: &QuantizedMatrixManifest) -> Result<PackedTensor> {
        let path = checked_archive_member(&self.root, &matrix.file)?;
        let bytes = read_verified_member(&path, matrix.packed_bytes, &matrix.sha256)?;
        PackedTensor::from_bytes(&bytes)
            .with_context(|| format!("invalid quantized matrix `{}`", matrix.name))
    }

    pub fn load_floating(&self, name: &str) -> Result<FloatingTensorData> {
        let tensor = self
            .manifest
            .floating_tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("quantized archive has no floating tensor `{name}`"))?;
        self.load_floating_entry(tensor)
    }

    fn load_floating_entry(&self, tensor: &FloatingTensorManifest) -> Result<FloatingTensorData> {
        let path = checked_archive_member(&self.root, &tensor.file)?;
        Ok(FloatingTensorData {
            name: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            bytes: read_verified_member(&path, tensor.bytes, &tensor.sha256)?,
        })
    }

    pub fn load_into<B: QuantizedModelBackend>(&self, backend: &mut B) -> Result<()> {
        if let Err(error) = backend.begin_archive_load(&self.manifest) {
            return match backend.rollback_archive_load() {
                Ok(()) => Err(error.context("failed to begin quantized archive load")),
                Err(rollback) => bail!(
                    "failed to begin quantized archive load: {error:#}; rollback also failed: {rollback:#}"
                ),
            };
        }
        let loaded = (|| {
            for matrix in &self.manifest.matrices {
                let packed = self.load_matrix_entry(matrix)?;
                backend.load_quantized_matrix(&matrix.name, &packed)?;
            }
            for tensor in &self.manifest.floating_tensors {
                backend.load_floating_tensor(&self.load_floating_entry(tensor)?)?;
            }
            backend.commit_archive_load()
        })();
        if let Err(error) = loaded {
            return match backend.rollback_archive_load() {
                Ok(()) => Err(error.context("failed to install quantized archive")),
                Err(rollback) => bail!(
                    "failed to install quantized archive: {error:#}; rollback also failed: {rollback:#}"
                ),
            };
        }
        Ok(())
    }
}

/// Export a complete weight archive from a safetensors checkpoint. Every
/// matrix selected by the recipe is encoded as HQUANT; norms, scalar state,
/// unsupported dtypes, and explicit opt-outs are preserved byte-for-byte.
/// Publication is one directory rename, so readers never observe a partial
/// manifest.
pub fn export_safetensors_archive(
    checkpoint: &Path,
    output: &Path,
    recipe: &QuantizationRecipe,
) -> Result<QuantizationManifest> {
    recipe.validate()?;
    let mut source = StreamingSafetensors::open(checkpoint, "source checkpoint")?;
    let source_hash = source.sha256().to_owned();
    if output.exists() {
        let archive = QuantizedArchive::open(output).with_context(|| {
            format!(
                "existing quantization output {} is not a valid archive",
                output.display()
            )
        })?;
        ensure!(
            archive.manifest.base_checkpoint_hash == source_hash
                && archive.manifest.recipe == *recipe,
            "existing quantization output does not match this checkpoint and recipe"
        );
        // Unlike trainer resume, this standalone API has no externally sealed
        // candidate receipt. Reproduce an existing output before accepting it
        // rather than treating its mutable manifest as provenance.
        archive
            .verify_source_safetensors(&mut source)
            .context("existing quantization output does not reproduce its source checkpoint")?;
        return Ok(archive.manifest.clone());
    }
    ensure!(
        !source.tensors.is_empty(),
        "safetensors checkpoint contains no tensors"
    );

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = output
        .file_name()
        .context("quantization output has no file name")?;
    let temporary = create_unique_archive_directory(parent, file_name)?;
    fs::create_dir(temporary.join("quantized"))?;
    fs::create_dir(temporary.join("floating"))?;

    let export = (|| {
        let mut matrices = Vec::new();
        let mut floating_tensors = Vec::new();
        let tensor_entries = source.tensor_entries();
        for (index, (name, tensor)) in tensor_entries.into_iter().enumerate() {
            let elements = checked_elements(&tensor.shape)?;
            let elements_u64 =
                u64::try_from(elements).context("tensor element count exceeds u64")?;
            let identifier = format!("{index:06}-{}", safe_file_component(&name));
            let tensor_bytes = source.read_tensor(&tensor)?;
            if should_quantize(&name, &tensor.shape, tensor.dtype, recipe) {
                let values = tensor_as_f32(tensor.dtype, &tensor_bytes)?;
                ensure!(
                    values.len() == elements,
                    "tensor `{name}` data length does not match its shape"
                );
                let packed = quantize_tensor(
                    &values,
                    tensor.shape.clone(),
                    recipe.format,
                    recipe.group_size,
                )?;
                let decoded = packed.decode()?;
                let (mean_squared_error, maximum_absolute_error) =
                    quantization_error(&values, &decoded);
                let relative = PathBuf::from("quantized").join(format!("{identifier}.hquant"));
                let bytes = packed.to_bytes()?;
                write_file_synced(&temporary.join(&relative), &bytes)?;
                matrices.push(QuantizedMatrixManifest {
                    name: name.clone(),
                    shape: tensor.shape.clone(),
                    elements: elements_u64,
                    format: recipe.format,
                    file: path_to_manifest(&relative)?,
                    packed_bytes: u64::try_from(bytes.len())
                        .context("packed tensor size exceeds u64")?,
                    sha256: sha256_identity(&bytes),
                    mean_squared_error,
                    maximum_absolute_error,
                });
            } else {
                let relative = PathBuf::from("floating").join(format!("{identifier}.bin"));
                write_file_synced(&temporary.join(&relative), &tensor_bytes)?;
                floating_tensors.push(FloatingTensorManifest {
                    name: name.clone(),
                    dtype: format!("{:?}", tensor.dtype),
                    shape: tensor.shape.clone(),
                    elements: elements_u64,
                    file: path_to_manifest(&relative)?,
                    bytes: u64::try_from(tensor_bytes.len())
                        .context("floating tensor size exceeds u64")?,
                    sha256: sha256_identity(&tensor_bytes),
                });
            }
        }
        source.ensure_unchanged()?;
        ensure!(
            !matrices.is_empty(),
            "checkpoint contains no eligible floating-point matrices"
        );
        let manifest = QuantizationManifest {
            version: ARCHIVE_VERSION,
            base_checkpoint_hash: source_hash.clone(),
            recipe: recipe.clone(),
            matrices,
            floating_tensors,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        write_file_synced(&temporary.join("manifest.json"), &manifest_bytes)?;
        fs::File::open(temporary.join("quantized"))?.sync_all()?;
        fs::File::open(temporary.join("floating"))?.sync_all()?;
        fs::File::open(&temporary)?.sync_all()?;
        fs::rename(&temporary, output)
            .with_context(|| format!("failed to publish {}", output.display()))?;
        fs::File::open(parent)?.sync_all()?;
        // Verify the exact manifest bytes, then validate every referenced file
        // from its final path before reporting publication. Comparing parsed
        // structs is incorrect here because serde_json may round diagnostic
        // floating-point values to a neighboring representation.
        ensure!(
            read_regular_file_bounded(
                &output.join("manifest.json"),
                MAX_QUANTIZATION_MANIFEST_BYTES,
                "published quantization manifest",
            )? == manifest_bytes,
            "published quantization manifest bytes changed during validation"
        );
        QuantizedArchive::open(output)?
            .verify_source_checkpoint(checkpoint)
            .context("published quantization archive does not reproduce its source checkpoint")?;
        Ok(manifest)
    })();
    if export.is_err() && temporary.exists() {
        // This directory was created by this invocation and has never been
        // published; removing it cannot affect an existing artifact.
        let _ = fs::remove_dir_all(&temporary);
    }
    export
}

fn should_quantize(name: &str, shape: &[usize], dtype: Dtype, recipe: &QuantizationRecipe) -> bool {
    if shape.len() < 2
        || shape.contains(&0)
        || !matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
    {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    let embedding = lower.contains("embedding") || lower.contains("token_embed");
    let lm_head = lower.contains("lm_head") || lower.contains("output_projection");
    (!embedding || recipe.quantize_embeddings) && (!lm_head || recipe.quantize_lm_head)
}

fn tensor_as_f32(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            ensure!(
                data.len().is_multiple_of(4),
                "F32 tensor byte length is invalid"
            );
            Ok(data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect())
        }
        Dtype::F16 => {
            ensure!(
                data.len().is_multiple_of(2),
                "F16 tensor byte length is invalid"
            );
            Ok(data
                .chunks_exact(2)
                .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            ensure!(
                data.len().is_multiple_of(2),
                "BF16 tensor byte length is invalid"
            );
            Ok(data
                .chunks_exact(2)
                .map(|bytes| {
                    bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()
                })
                .collect())
        }
        _ => bail!("dtype {dtype:?} is not eligible for ultra-quantization"),
    }
}

fn quantization_error(source: &[f32], decoded: &[f32]) -> (f64, f32) {
    let mut squared = 0.0f64;
    let mut maximum = 0.0f32;
    for (&source, &decoded) in source.iter().zip(decoded) {
        let error = (source - decoded).abs();
        squared += f64::from(error) * f64::from(error);
        maximum = maximum.max(error);
    }
    (squared / source.len() as f64, maximum)
}

fn safe_file_component(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn path_to_manifest(path: &Path) -> Result<String> {
    path.to_str()
        .map(|path| path.replace('\\', "/"))
        .context("archive path is not UTF-8")
}

fn create_unique_archive_directory(
    parent: &Path,
    output_name: &std::ffi::OsStr,
) -> Result<PathBuf> {
    for attempt in 0..1024u32 {
        let mut name = std::ffi::OsString::from(".");
        name.push(output_name);
        name.push(format!(".tmp-{}-{attempt}", std::process::id()));
        let candidate = parent.join(name);
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to create quantization temporary directory {}",
                        candidate.display()
                    )
                });
            }
        }
    }
    bail!("could not allocate a quantization temporary directory")
}

fn validate_member_identity(
    name: &str,
    file: &str,
    names: &mut BTreeSet<String>,
    files: &mut BTreeSet<String>,
) -> Result<()> {
    ensure!(!name.trim().is_empty(), "archive tensor name is empty");
    ensure!(
        name.len() <= MAX_TENSOR_NAME_BYTES,
        "archive tensor name exceeds its {MAX_TENSOR_NAME_BYTES}-byte limit"
    );
    ensure!(
        names.insert(name.to_owned()),
        "archive repeats tensor name `{name}`"
    );
    ensure!(
        !file.trim().is_empty(),
        "archive tensor `{name}` has no file"
    );
    ensure!(
        file.len() <= MAX_ARCHIVE_MEMBER_PATH_BYTES,
        "archive member path exceeds its {MAX_ARCHIVE_MEMBER_PATH_BYTES}-byte limit"
    );
    ensure!(
        files.insert(file.to_owned()),
        "archive member file `{file}` is referenced more than once"
    );
    Ok(())
}

fn checked_archive_member(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "archive member path must be a normalized relative path"
    );
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            unreachable!("components were validated above")
        };
        current.push(component);
        let metadata = fs::symlink_metadata(&current)
            .with_context(|| format!("failed to inspect archive member {}", current.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "archive member path contains a symlink: {}",
            current.display()
        );
    }
    let metadata = fs::symlink_metadata(&current)?;
    ensure!(
        metadata.is_file(),
        "archive member is not a regular file: {}",
        current.display()
    );
    Ok(current)
}

/// Reject files which are not authenticated by the manifest. Without this
/// closed inventory, two directory trees with the same reported content hash
/// could carry different unverified payloads, and an unnoticed temporary file
/// could later be mistaken for part of the immutable archive.
fn validate_archive_inventory(root: &Path, members: &BTreeSet<String>) -> Result<()> {
    fn visit(
        root: &Path,
        directory: &Path,
        depth: usize,
        entries_seen: &mut usize,
        files: &mut BTreeSet<String>,
    ) -> Result<()> {
        ensure!(
            depth <= MAX_QUANTIZATION_ARCHIVE_DEPTH,
            "quantized archive exceeds its {MAX_QUANTIZATION_ARCHIVE_DEPTH}-directory depth limit"
        );
        let entries = fs::read_dir(directory)
            .with_context(|| format!("failed to read archive directory {}", directory.display()))?;
        for entry in entries {
            let entry = entry?;
            *entries_seen = entries_seen
                .checked_add(1)
                .context("quantized archive entry count overflow")?;
            ensure!(
                *entries_seen
                    <= MAX_QUANTIZATION_ARCHIVE_MEMBERS + MAX_QUANTIZATION_ARCHIVE_DIRECTORIES + 1,
                "quantized archive exceeds its bounded file/directory inventory"
            );
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "quantized archive contains a symlink: {}",
                path.display()
            );
            if metadata.is_dir() {
                visit(root, &path, depth + 1, entries_seen, files)?;
                continue;
            }
            ensure!(
                metadata.is_file(),
                "quantized archive contains a non-file member: {}",
                path.display()
            );
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .context("quantized archive member path is not UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            ensure!(
                files.insert(relative.clone()),
                "quantized archive repeats member path `{relative}`"
            );
        }
        Ok(())
    }

    let mut expected = members.clone();
    expected.insert("manifest.json".to_owned());
    let mut actual = BTreeSet::new();
    let mut entries_seen = 0;
    visit(root, root, 0, &mut entries_seen, &mut actual)?;
    ensure!(
        actual == expected,
        "quantized archive file inventory differs from its manifest: expected {expected:?}, got {actual:?}"
    );
    Ok(())
}

fn read_verified_member(
    path: &Path,
    declared_bytes: u64,
    declared_sha256: &str,
) -> Result<Vec<u8>> {
    ensure!(
        declared_bytes <= MAX_QUANTIZATION_MEMBER_BYTES,
        "archive member {} exceeds its {MAX_QUANTIZATION_MEMBER_BYTES}-byte limit",
        path.display()
    );
    validate_sha256_identity(declared_sha256, "quantization archive member identity")
        .with_context(|| {
            format!(
                "archive member {} has an invalid SHA-256 label",
                path.display()
            )
        })?;
    let bytes = read_regular_file_exact(path, declared_bytes, "archive member")?;
    ensure!(
        sha256_identity(&bytes) == declared_sha256,
        "archive member {} SHA-256 mismatch",
        path.display()
    );
    Ok(bytes)
}

fn packed_tensor_storage_bytes(
    shape: &[usize],
    format: UltraQuantFormat,
    group_size: usize,
) -> Result<u64> {
    ensure!(group_size == BONSAI_GROUP_SIZE, "packed tensor is not g128");
    let elements = checked_elements(shape)?;
    ensure!(elements > 0, "packed tensor is empty");
    let rank = u64::try_from(shape.len()).context("packed tensor rank exceeds u64")?;
    ensure!(rank <= u64::from(u8::MAX), "packed tensor rank exceeds u8");
    let groups = elements.div_ceil(group_size);
    let codes_per_group = match format {
        UltraQuantFormat::BinaryG128 => group_size / 8,
        UltraQuantFormat::TernaryG128 => group_size / 4,
        UltraQuantFormat::TernaryEntropyG128 => group_size.div_ceil(5),
    };
    let payload = groups
        .checked_mul(2)
        .and_then(|scales| {
            groups
                .checked_mul(codes_per_group)
                .and_then(|codes| scales.checked_add(codes))
        })
        .context("packed tensor payload size overflows usize")?;
    let payload = u64::try_from(payload).context("packed tensor payload exceeds u64")?;
    // Fixed header/checksum plus one u64 for every dimension.
    38_u64
        .checked_add(
            rank.checked_mul(8)
                .context("packed tensor shape header overflows u64")?,
        )
        .and_then(|bytes| bytes.checked_add(payload))
        .context("packed tensor storage size overflows u64")
}

fn tensor_storage_bytes(dtype: &str, elements: usize) -> Result<u64> {
    let bits = match dtype {
        "F4" => 4u64,
        "F6_E2M3" | "F6_E3M2" => 6,
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" => 8,
        "I16" | "U16" | "F16" | "BF16" => 16,
        "I32" | "U32" | "F32" => 32,
        "C64" | "F64" | "I64" | "U64" => 64,
        _ => bail!("unsupported floating tensor dtype `{dtype}` in quantized archive"),
    };
    let elements = u64::try_from(elements).context("tensor element count exceeds u64")?;
    elements
        .checked_mul(bits)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .context("tensor storage byte count overflows u64")
}

struct StreamingSafetensors {
    file: fs::File,
    path: PathBuf,
    label: String,
    identity: StableIdentity,
    sha256: String,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

impl StreamingSafetensors {
    fn open(path: &Path, label: &str) -> Result<Self> {
        let mut file = open_regular_file(path, label)?;
        let metadata = file
            .metadata()
            .with_context(|| format!("failed to inspect opened {label} {}", path.display()))?;
        ensure!(metadata.is_file(), "opened {label} is not a regular file");
        let identity = StableIdentity::from_metadata(&metadata);
        let (_, digest, _) = hash_open_file(&mut file, None, label)?;
        file.seek(SeekFrom::Start(0))?;

        let mut length_bytes = [0_u8; 8];
        file.read_exact(&mut length_bytes)
            .with_context(|| format!("{label} has a truncated safetensors header"))?;
        let header_bytes = u64::from_le_bytes(length_bytes);
        ensure!(
            header_bytes <= MAX_SAFETENSORS_HEADER_BYTES,
            "{label} safetensors header exceeds its {MAX_SAFETENSORS_HEADER_BYTES}-byte limit"
        );
        let header_len = usize::try_from(header_bytes)
            .with_context(|| format!("{label} safetensors header exceeds this address space"))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_len)
            .with_context(|| format!("reserving {label} safetensors header"))?;
        header.resize(header_len, 0);
        file.read_exact(&mut header)
            .with_context(|| format!("{label} safetensors header is truncated"))?;
        let parsed: SafeTensorMetadata = serde_json::from_slice(&header)
            .with_context(|| format!("invalid {label} safetensors metadata"))?;
        let data_start = 8_u64
            .checked_add(header_bytes)
            .context("safetensors data offset overflows u64")?;
        let data_bytes =
            u64::try_from(parsed.data_len()).context("safetensors payload length exceeds u64")?;
        ensure!(
            data_start.checked_add(data_bytes) == Some(metadata.len()),
            "{label} safetensors metadata does not cover the exact file"
        );
        let tensors = parsed
            .tensors()
            .into_iter()
            .map(|(name, info)| (name, info.clone()))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            !tensors.is_empty() && tensors.len() <= MAX_QUANTIZATION_ARCHIVE_MEMBERS,
            "{label} tensor inventory must contain 1..={MAX_QUANTIZATION_ARCHIVE_MEMBERS} entries"
        );
        for (name, tensor) in &tensors {
            ensure!(
                !name.trim().is_empty() && name.len() <= MAX_TENSOR_NAME_BYTES,
                "{label} contains an invalid or oversized tensor name"
            );
            ensure!(
                tensor.shape.len() <= MAX_TENSOR_RANK,
                "{label} tensor `{name}` has unsupported rank {}",
                tensor.shape.len()
            );
        }
        Ok(Self {
            file,
            path: path.to_owned(),
            label: label.to_owned(),
            identity,
            sha256: format!("sha256:{digest}"),
            data_start,
            tensors,
        })
    }

    fn sha256(&self) -> &str {
        &self.sha256
    }

    fn tensor_entries(&self) -> Vec<(String, TensorInfo)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect()
    }

    fn read_tensor(&mut self, tensor: &TensorInfo) -> Result<Vec<u8>> {
        let bytes = tensor
            .data_offsets
            .1
            .checked_sub(tensor.data_offsets.0)
            .context("safetensors tensor offsets move backwards")?;
        let bytes_u64 = u64::try_from(bytes).context("safetensors tensor size exceeds u64")?;
        ensure!(
            bytes_u64 <= MAX_QUANTIZATION_MEMBER_BYTES,
            "source tensor exceeds its {MAX_QUANTIZATION_MEMBER_BYTES}-byte materialization limit"
        );
        let offset = self
            .data_start
            .checked_add(
                u64::try_from(tensor.data_offsets.0)
                    .context("safetensors tensor offset exceeds u64")?,
            )
            .context("safetensors tensor file offset overflows u64")?;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)
            .context("reserving source tensor buffer")?;
        data.resize(bytes, 0);
        self.file.read_exact(&mut data).with_context(|| {
            format!(
                "source tensor payload is truncated in {}",
                self.path.display()
            )
        })?;
        Ok(data)
    }

    fn ensure_unchanged(&mut self) -> Result<()> {
        ensure!(
            StableIdentity::from_metadata(&self.file.metadata()?) == self.identity,
            "{} {} changed while tensors were streamed",
            self.label,
            self.path.display()
        );
        let (_, digest, _) = hash_open_file(&mut self.file, None, &self.label)?;
        ensure!(
            format!("sha256:{digest}") == self.sha256,
            "{} {} changed while tensors were streamed",
            self.label,
            self.path.display()
        );
        Ok(())
    }
}

fn open_regular_file(path: &Path, label: &str) -> Result<fs::File> {
    let inspected = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        inspected.is_file() && !inspected.file_type().is_symlink(),
        "{label} {} must be a regular non-symlink file",
        path.display()
    );
    #[cfg(unix)]
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("failed to open {label} {}", path.display()))?;
    #[cfg(not(unix))]
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open {label} {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label} {}", path.display()))?;
    ensure!(
        StableIdentity::from_metadata(&opened) == StableIdentity::from_metadata(&inspected),
        "{label} {} changed while it was opened",
        path.display()
    );
    Ok(file)
}

fn read_regular_file_exact(path: &Path, expected: u64, label: &str) -> Result<Vec<u8>> {
    read_open_regular_file_exact(open_regular_file(path, label)?, path, expected, label)
}

fn read_open_regular_file_exact(
    mut file: fs::File,
    path: &Path,
    expected: u64,
    label: &str,
) -> Result<Vec<u8>> {
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label} {}", path.display()))?;
    ensure!(
        metadata.is_file() && metadata.len() == expected,
        "{label} {} is not a regular file of the declared size",
        path.display()
    );
    let capacity = usize::try_from(expected)
        .with_context(|| format!("{label} {} exceeds this address space", path.display()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .with_context(|| format!("reserving {expected} bytes for {label} {}", path.display()))?;
    let mut buffer = [0_u8; 1024 * 1024];
    while bytes.len() < capacity {
        let remaining = capacity - bytes.len();
        let chunk = remaining.min(buffer.len());
        let read = file
            .read(&mut buffer[..chunk])
            .with_context(|| format!("failed to read opened {label} {}", path.display()))?;
        ensure!(
            read > 0,
            "{label} {} was truncated while read",
            path.display()
        );
        bytes.extend_from_slice(&buffer[..read]);
    }
    let mut tail = [0_u8; 1];
    ensure!(
        file.read(&mut tail)
            .with_context(|| format!("failed to finish reading {label} {}", path.display()))?
            == 0,
        "{label} {} grew while read",
        path.display()
    );
    Ok(bytes)
}

fn read_regular_file_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let file = open_regular_file(path, label)?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect opened {label} {}", path.display()))?;
    ensure!(metadata.is_file(), "{label} is not a regular file");
    ensure!(
        metadata.len() <= maximum,
        "{label} {} exceeds the {maximum}-byte limit",
        path.display()
    );
    // Consume the already-opened descriptor so pathname replacement cannot
    // change either the allocation bound or the returned bytes.
    read_open_regular_file_exact(file, path, metadata.len(), label)
}

fn write_file_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut value = [0; 1];
    reader.read_exact(&mut value)?;
    Ok(value[0])
}

fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut value = [0; 2];
    reader.read_exact(&mut value)?;
    Ok(u16::from_le_bytes(value))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut value = [0; 8];
    reader.read_exact(&mut value)?;
    Ok(u64::from_le_bytes(value))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> Vec<f32> {
        (0..BONSAI_GROUP_SIZE * 2)
            .map(|index| ((index as f32 * 0.37).sin() * 2.0) + (index % 7) as f32 * 0.01)
            .collect()
    }

    #[test]
    fn archive_inventory_rejects_excessive_directory_depth_without_recursive_overflow() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("manifest.json"), b"{}").unwrap();
        let mut nested = directory.path().to_path_buf();
        for depth in 0..=MAX_QUANTIZATION_ARCHIVE_DEPTH {
            nested.push(format!("level-{depth}"));
            fs::create_dir(&nested).unwrap();
        }
        let error = validate_archive_inventory(directory.path(), &BTreeSet::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("depth limit"), "{error}");
    }

    #[test]
    fn device_fake_quant_matches_archive_targets_and_uses_ste() {
        let device = Device::default().autodiff();
        let values = (0..259)
            .map(|index| ((index as f32 * 0.173).sin() * 1.7) + (index % 11) as f32 * 0.003)
            .collect::<Vec<_>>();
        for format in [
            UltraQuantFormat::BinaryG128,
            UltraQuantFormat::TernaryG128,
            UltraQuantFormat::TernaryEntropyG128,
        ] {
            let source =
                Tensor::<1>::from_data(TensorData::new(values.clone(), [values.len()]), &device)
                    .require_grad();
            let output = fake_quantize_tensor(source.clone(), format);
            let actual = output
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let expected = fake_quantize(&values, format).unwrap();
            let maximum_error = actual
                .iter()
                .zip(&expected)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            // Archive serialization rounds each group scale to FP16; the QAT
            // path deliberately keeps that scale in the training dtype.
            assert!(maximum_error < 2e-3, "{format:?}: {maximum_error}");

            let mut gradients = output.sum().backward();
            let gradient = source.grad_remove(&mut gradients).unwrap();
            let gradient = gradient
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            assert!(gradient.iter().all(|value| (*value - 1.0).abs() < 1e-6));
        }
    }

    #[test]
    fn transformer_qat_clone_keeps_parameter_ids_and_supports_an_accumulation_window() {
        let device = Device::default().autodiff();
        let mut config = hermes_llm::get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 2;
        config.max_seq_len = 8;
        config.embeddings.tie_weights = false;
        if let Some(pattern) = &mut config.pattern {
            for block in pattern {
                block.attention.num_heads = Some(2);
                block.attention.num_kv_heads = Some(1);
                block.attention.head_dim = Some(4);
                block.ffn.hidden_dim = Some(16);
                block.dropout = 0.0;
                block.attention.dropout = 0.0;
                block.ffn.dropout = 0.0;
            }
        }
        let master = Transformer::new(&config, &device).unwrap();
        let before = burn::module::list_param_ids(&master);
        let (staged, tensors) =
            fake_quantized_transformer(&master, UltraQuantFormat::BinaryG128, true, true).unwrap();
        assert!(tensors > 0);
        assert_eq!(burn::module::list_param_ids(&staged), before);

        // A staged leaf model is reused for every microbatch until the master
        // update. Repeated backwards must keep producing gradients for the
        // original parameter IDs; otherwise window-level staging would drop
        // all but the first microbatch.
        for offset in 0..2 {
            let input =
                Tensor::<2, Int>::from_data([[1 + offset, 2 + offset, 3 + offset, 4]], &device);
            let target = Tensor::<2, Int>::from_data([[2 + offset, 3 + offset, 4, 5]], &device);
            let mut gradients = staged.forward_loss(input, target).backward();
            let gradients = burn_optim::GradientsParams::from_module(&mut gradients, &staged);
            assert!(!gradients.is_empty());
            let missing_muon = master
                .muon_parameter_ids()
                .into_iter()
                .filter(|id| gradients.get::<2>(*id).is_none())
                .collect::<Vec<_>>();
            assert!(
                missing_muon.is_empty(),
                "fake-quantized forward omitted Muon gradients: {missing_muon:?}"
            );
        }
    }

    #[test]
    fn transformer_qat_and_archive_select_the_exact_same_matrix_names() {
        use burn::module::AutodiffModule;

        let device = Device::default().autodiff();
        let mut config = hermes_llm::get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 2;
        config.max_seq_len = 8;
        config.embeddings.tie_weights = false;
        if let Some(pattern) = &mut config.pattern {
            for block in pattern {
                block.attention.num_heads = Some(2);
                block.attention.num_kv_heads = Some(1);
                block.attention.head_dim = Some(4);
                block.ffn.hidden_dim = Some(16);
                block.dropout = 0.0;
                block.attention.dropout = 0.0;
                block.ffn.dropout = 0.0;
            }
        }
        let model = Transformer::new(&config, &device).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("weights.safetensors");
        hermes_llm::save_safetensors(&model.valid(), &checkpoint).unwrap();
        for (ordinal, (quantize_embeddings, quantize_lm_head)) in
            [(false, false), (false, true), (true, false), (true, true)]
                .into_iter()
                .enumerate()
        {
            let expected = model
                .ultra_quant_parameter_names(quantize_embeddings, quantize_lm_head)
                .unwrap()
                .into_iter()
                .collect::<BTreeSet<_>>();
            let output = directory.path().join(format!("weights-{ordinal}.hquant"));
            let recipe = QuantizationRecipe {
                format: UltraQuantFormat::BinaryG128,
                group_size: BONSAI_GROUP_SIZE,
                fake_quant_start_step: 0,
                ternary_warmup_steps: 0,
                distillation_weight: 0.0,
                quantize_embeddings,
                quantize_lm_head,
            };
            let manifest = export_safetensors_archive(&checkpoint, &output, &recipe).unwrap();
            let actual = manifest
                .matrices
                .into_iter()
                .map(|matrix| matrix.name)
                .collect::<BTreeSet<_>>();
            assert_eq!(
                actual, expected,
                "QAT ParamIds and archive names differ for embeddings={quantize_embeddings}, lm_head={quantize_lm_head}"
            );
        }

        config.embeddings.tie_weights = true;
        let tied = Transformer::new(&config, &device).unwrap();
        let tied_checkpoint = directory.path().join("tied-weights.safetensors");
        hermes_llm::save_safetensors(&tied.valid(), &tied_checkpoint).unwrap();
        for (ordinal, quantize_tied_weight) in [false, true].into_iter().enumerate() {
            let expected = tied
                .ultra_quant_parameter_names(quantize_tied_weight, quantize_tied_weight)
                .unwrap()
                .into_iter()
                .collect::<BTreeSet<_>>();
            let output = directory
                .path()
                .join(format!("tied-weights-{ordinal}.hquant"));
            let recipe = QuantizationRecipe {
                format: UltraQuantFormat::BinaryG128,
                group_size: BONSAI_GROUP_SIZE,
                fake_quant_start_step: 0,
                ternary_warmup_steps: 0,
                distillation_weight: 0.0,
                quantize_embeddings: quantize_tied_weight,
                quantize_lm_head: quantize_tied_weight,
            };
            let manifest = export_safetensors_archive(&tied_checkpoint, &output, &recipe).unwrap();
            let actual = manifest
                .matrices
                .into_iter()
                .map(|matrix| matrix.name)
                .collect::<BTreeSet<_>>();
            assert_eq!(
                actual, expected,
                "tied QAT/archive names differ for quantize_tied_weight={quantize_tied_weight}"
            );
        }
        for (quantize_embeddings, quantize_lm_head) in [(false, true), (true, false)] {
            let error = tied
                .ultra_quant_parameter_names(quantize_embeddings, quantize_lm_head)
                .unwrap_err()
                .to_string();
            assert!(error.contains("tied embedding/output weights"), "{error}");
        }
    }

    #[test]
    fn binary_g128_has_prism_public_storage_rate_and_roundtrips() {
        let packed = quantize_tensor(
            &weights(),
            vec![2, BONSAI_GROUP_SIZE],
            UltraQuantFormat::BinaryG128,
            BONSAI_GROUP_SIZE,
        )
        .unwrap();
        assert_eq!(packed.codes.len(), 32);
        assert_eq!(packed.scales.len(), 2);
        assert!((packed.bits_per_weight().unwrap() - 1.125).abs() < 1e-9);
        let decoded = packed.decode().unwrap();
        assert_eq!(decoded.len(), weights().len());
        assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn malformed_public_packed_tensor_geometry_returns_errors_without_panicking() {
        let malformed = PackedTensor {
            format: UltraQuantFormat::BinaryG128,
            shape: vec![usize::MAX, 2],
            group_size: BONSAI_GROUP_SIZE,
            scales: Vec::new(),
            codes: Vec::new(),
        };

        for error in [
            malformed.elements().unwrap_err(),
            malformed.bits_per_weight().unwrap_err(),
            malformed.decode().unwrap_err(),
            malformed.to_bytes().unwrap_err(),
            malformed.matrix_vector(&[0.0, 0.0]).unwrap_err(),
        ] {
            assert!(
                format!("{error:#}").contains("element count overflows"),
                "{error:#}"
            );
        }

        let invalid_rank = PackedTensor {
            shape: Vec::new(),
            ..malformed
        };
        let error = invalid_rank.bits_per_weight().unwrap_err();
        assert!(format!("{error:#}").contains("rank must be"), "{error:#}");
    }

    #[test]
    fn ternary_codecs_decode_identically_and_entropy_pack_is_smaller() {
        let dense = quantize_tensor(
            &weights(),
            vec![2, BONSAI_GROUP_SIZE],
            UltraQuantFormat::TernaryG128,
            BONSAI_GROUP_SIZE,
        )
        .unwrap();
        let entropy = quantize_tensor(
            &weights(),
            vec![2, BONSAI_GROUP_SIZE],
            UltraQuantFormat::TernaryEntropyG128,
            BONSAI_GROUP_SIZE,
        )
        .unwrap();
        assert_eq!(dense.decode().unwrap(), entropy.decode().unwrap());
        assert!(entropy.codes.len() < dense.codes.len());
        assert_eq!(dense.bits_per_weight().unwrap(), 2.125);
        assert_eq!(entropy.bits_per_weight().unwrap(), 1.75);
    }

    #[test]
    fn final_partial_group_is_padded_only_in_storage() {
        let values = (0..137)
            .map(|index| index as f32 - 70.0)
            .collect::<Vec<_>>();
        for format in [
            UltraQuantFormat::BinaryG128,
            UltraQuantFormat::TernaryG128,
            UltraQuantFormat::TernaryEntropyG128,
        ] {
            let packed = quantize_tensor(&values, vec![137], format, BONSAI_GROUP_SIZE).unwrap();
            assert_eq!(packed.scales.len(), 2);
            assert_eq!(packed.decode().unwrap().len(), values.len());
            assert_eq!(
                PackedTensor::from_bytes(&packed.to_bytes().unwrap()).unwrap(),
                packed
            );
        }
    }

    #[test]
    fn direct_packed_linear_matches_decode_reference() {
        let values = weights();
        let input = (0..BONSAI_GROUP_SIZE)
            .map(|index| (index as f32 * 0.2).cos())
            .collect::<Vec<_>>();
        for format in [
            UltraQuantFormat::BinaryG128,
            UltraQuantFormat::TernaryG128,
            UltraQuantFormat::TernaryEntropyG128,
        ] {
            let packed = quantize_tensor(
                &values,
                vec![2, BONSAI_GROUP_SIZE],
                format,
                BONSAI_GROUP_SIZE,
            )
            .unwrap();
            let decoded = packed.decode().unwrap();
            let expected = decoded
                .chunks_exact(BONSAI_GROUP_SIZE)
                .map(|row| row.iter().zip(&input).map(|(a, b)| a * b).sum::<f32>())
                .collect::<Vec<_>>();
            let actual = packed.matrix_vector(&input).unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn serialized_codec_detects_corruption() {
        let packed = quantize_tensor(
            &weights(),
            vec![2, BONSAI_GROUP_SIZE],
            UltraQuantFormat::BinaryG128,
            BONSAI_GROUP_SIZE,
        )
        .unwrap();
        let bytes = packed.to_bytes().unwrap();
        assert_eq!(PackedTensor::from_bytes(&bytes).unwrap(), packed);
        let mut corrupt = bytes;
        corrupt[20] ^= 1;
        assert!(PackedTensor::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn codec_rejects_non_canonical_scales_and_padding() {
        let values = (0..137)
            .map(|index| index as f32 - 70.0)
            .collect::<Vec<_>>();
        for format in [
            UltraQuantFormat::BinaryG128,
            UltraQuantFormat::TernaryG128,
            UltraQuantFormat::TernaryEntropyG128,
        ] {
            let packed = quantize_tensor(&values, vec![137], format, BONSAI_GROUP_SIZE).unwrap();

            let mut negative_zero = packed.clone();
            negative_zero.scales[0] = f16::from_f32(-0.0).to_bits();
            assert!(negative_zero.to_bytes().is_err(), "{format:?}");

            let mut padded = packed;
            let bytes_per_group = padded.codes.len() / padded.scales.len();
            let final_group = bytes_per_group;
            match format {
                UltraQuantFormat::BinaryG128 => padded.codes[final_group + 1] |= 1 << 7,
                UltraQuantFormat::TernaryG128 => padded.codes[final_group + 2] |= 1 << 2,
                UltraQuantFormat::TernaryEntropyG128 => {
                    padded.codes[final_group + 1] += 3u8.pow(4);
                }
            }
            assert!(padded.to_bytes().is_err(), "{format:?}");
        }
    }

    #[test]
    fn atomic_file_roundtrip() {
        let packed = quantize_tensor(
            &weights(),
            vec![2, BONSAI_GROUP_SIZE],
            UltraQuantFormat::TernaryEntropyG128,
            BONSAI_GROUP_SIZE,
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("matrix.hquant");
        packed.save_atomic(&path).unwrap();
        assert_eq!(PackedTensor::load(&path).unwrap(), packed);
    }

    #[test]
    fn packed_tensor_load_rejects_oversized_file_before_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.hquant");
        let file = fs::File::create(&path).unwrap();
        file.set_len(MAX_QUANTIZATION_MEMBER_BYTES + 1).unwrap();
        drop(file);

        let error = PackedTensor::load(&path).unwrap_err().to_string();
        assert!(error.contains("byte limit"), "{error}");
    }

    #[test]
    fn progressive_binary_recipe_uses_ternary_warmup() {
        let recipe = QuantizationRecipe {
            format: UltraQuantFormat::BinaryG128,
            group_size: BONSAI_GROUP_SIZE,
            fake_quant_start_step: 100,
            ternary_warmup_steps: 50,
            distillation_weight: 1.0,
            quantize_embeddings: true,
            quantize_lm_head: true,
        };
        recipe.validate().unwrap();
        assert_eq!(recipe.forward_format(99), None);
        assert_eq!(
            recipe.forward_format(100),
            Some(UltraQuantFormat::TernaryG128)
        );
        assert_eq!(
            recipe.forward_format(149),
            Some(UltraQuantFormat::TernaryG128)
        );
        assert_eq!(
            recipe.forward_format(150),
            Some(UltraQuantFormat::BinaryG128)
        );
    }

    #[test]
    fn workflow_quantization_bridge_preserves_schedule_and_training_mode() {
        let config: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "format": "binary_g128",
            "warmup_format": "ternary_entropy_g128",
            "group_size": 128,
            "start_step": 10,
            "end_step": 30,
            "embeddings": false,
            "lm_head": true,
            "training": {
                "type": "qat",
                "warmup_steps": 5,
                "straight_through": true
            }
        }))
        .unwrap();
        let plan = WorkflowQuantizationPlan::from_workflow(&config).unwrap();
        assert_eq!(plan.recipe.format, UltraQuantFormat::BinaryG128);
        assert!(!plan.recipe.quantize_embeddings);
        assert_eq!(plan.format_at(9), None);
        assert_eq!(
            plan.format_at(10),
            Some(UltraQuantFormat::TernaryEntropyG128)
        );
        assert_eq!(
            plan.format_at(14),
            Some(UltraQuantFormat::TernaryEntropyG128)
        );
        assert_eq!(plan.format_at(15), Some(UltraQuantFormat::BinaryG128));
        assert_eq!(plan.format_at(30), None);
        assert_eq!(plan, WorkflowQuantizationPlan::try_from(&config).unwrap());

        let mut unsupported = config;
        unsupported.training = QuantizationTraining::Qat {
            warmup_steps: 5,
            straight_through: false,
        };
        assert!(WorkflowQuantizationPlan::from_workflow(&unsupported).is_err());
    }

    #[test]
    fn workflow_qat_can_warm_up_in_full_precision() {
        let config: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "format": "ternary_g128",
            "start_step": 100,
            "training": {"type": "qat", "warmup_steps": 3}
        }))
        .unwrap();
        let plan = WorkflowQuantizationPlan::from_workflow(&config).unwrap();
        assert_eq!(plan.format_at(100), None);
        assert_eq!(plan.format_at(102), None);
        assert_eq!(plan.format_at(103), Some(UltraQuantFormat::TernaryG128));
        assert_eq!(plan.recipe.fake_quant_start_step, 103);
    }

    #[test]
    fn target_format_must_intersect_the_concrete_phase_window() {
        let config: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "format": "binary_g128",
            "warmup_format": "ternary_g128",
            "start_step": 10,
            "end_step": 30,
            "training": {"type": "qat", "warmup_steps": 5}
        }))
        .unwrap();
        let plan = WorkflowQuantizationPlan::from_workflow(&config).unwrap();

        assert!(plan.validate_phase_window(0, 15).is_err());
        plan.validate_phase_window(0, 16).unwrap();
        plan.validate_phase_window(29, 1).unwrap();
        assert!(plan.validate_phase_window(30, 1).is_err());

        let delayed: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "format": "ternary_g128",
            "start_step": 100,
            "training": {"type": "qat", "warmup_steps": 3}
        }))
        .unwrap();
        let delayed = WorkflowQuantizationPlan::from_workflow(&delayed).unwrap();
        assert!(delayed.validate_phase_window(0, 100).is_err());
        assert!(delayed.validate_phase_window(100, 3).is_err());
        delayed.validate_phase_window(100, 4).unwrap();
    }

    #[test]
    fn workflow_distillation_bridge_keeps_teacher_controls() {
        let config: QuantizationConfig = serde_json::from_value(serde_json::json!({
            "format": "ternary_g128",
            "start_step": 4,
            "embeddings": true,
            "lm_head": false,
            "training": {
                "type": "distillation",
                "teacher_checkpoint": "/checkpoints/teacher.safetensors",
                "teacher_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "temperature": 2.5,
                "loss_weight": 0.75
            }
        }))
        .unwrap();
        let plan = WorkflowQuantizationPlan::from_workflow(&config).unwrap();
        assert_eq!(plan.recipe.distillation_weight, 0.75);
        assert!(!plan.recipe.quantize_lm_head);
        assert_eq!(
            QuantizationRecipe::from_workflow(&config).unwrap(),
            plan.recipe
        );
        let WorkflowQuantizationTraining::Distillation {
            teacher_checkpoint,
            teacher_sha256,
            temperature,
            loss_weight,
        } = plan.training
        else {
            panic!("expected distillation plan")
        };
        assert_eq!(
            teacher_checkpoint,
            Path::new("/checkpoints/teacher.safetensors")
        );
        assert_eq!(
            teacher_sha256,
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(temperature, 2.5);
        assert_eq!(loss_weight, 0.75);
    }

    #[test]
    fn true_average_bpw_includes_float_tail() {
        let manifest = QuantizationManifest {
            version: ARCHIVE_VERSION,
            base_checkpoint_hash: "abc".into(),
            recipe: QuantizationRecipe {
                format: UltraQuantFormat::BinaryG128,
                group_size: BONSAI_GROUP_SIZE,
                fake_quant_start_step: 0,
                ternary_warmup_steps: 0,
                distillation_weight: 0.0,
                quantize_embeddings: true,
                quantize_lm_head: true,
            },
            matrices: vec![QuantizedMatrixManifest {
                name: "weight".into(),
                shape: vec![1, 128],
                elements: 128,
                format: UltraQuantFormat::BinaryG128,
                file: "quantized/weight.hquant".into(),
                packed_bytes: 18,
                sha256: format!("sha256:{}", "0".repeat(64)),
                mean_squared_error: 0.1,
                maximum_absolute_error: 0.5,
            }],
            floating_tensors: vec![FloatingTensorManifest {
                name: "norm".into(),
                dtype: "F32".into(),
                shape: vec![1],
                elements: 1,
                file: "floating/norm.bin".into(),
                bytes: 4,
                sha256: format!("sha256:{}", "0".repeat(64)),
            }],
        };
        let expected = (18.0 * 8.0 + 32.0) / 129.0;
        assert!((manifest.true_average_bits_per_weight().unwrap() - expected).abs() < 1e-12);
    }

    fn export_source_verification_fixture(
        directory: &Path,
        name: &str,
    ) -> (PathBuf, PathBuf, QuantizationRecipe) {
        use safetensors::tensor::TensorView;

        let first = (0..137)
            .map(|index| (index as f32 * 0.1).sin())
            .collect::<Vec<_>>();
        let second = (0..256)
            .map(|index| (index as f32 * 0.07).cos())
            .collect::<Vec<_>>();
        let first_bytes = first
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let second_bytes = second
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let norm_bytes = [1.0f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let views = vec![
            (
                "layers.0.weight",
                TensorView::new(Dtype::F32, vec![1, 137], &first_bytes).unwrap(),
            ),
            (
                "layers.1.weight",
                TensorView::new(Dtype::F32, vec![2, 128], &second_bytes).unwrap(),
            ),
            (
                "layers.0.norm",
                TensorView::new(Dtype::F32, vec![2], &norm_bytes).unwrap(),
            ),
        ];
        let checkpoint = safetensors::serialize(views, None).unwrap();
        let checkpoint_path = directory.join(format!("{name}.safetensors"));
        let output = directory.join(format!("{name}.hquant"));
        fs::write(&checkpoint_path, checkpoint).unwrap();
        let recipe = QuantizationRecipe {
            format: UltraQuantFormat::BinaryG128,
            group_size: BONSAI_GROUP_SIZE,
            fake_quant_start_step: 0,
            ternary_warmup_steps: 0,
            distillation_weight: 1.0,
            quantize_embeddings: true,
            quantize_lm_head: true,
        };
        export_safetensors_archive(&checkpoint_path, &output, &recipe).unwrap();
        (checkpoint_path, output, recipe)
    }

    fn read_archive_manifest(output: &Path) -> QuantizationManifest {
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap()
    }

    fn write_archive_manifest(output: &Path, manifest: &QuantizationManifest) {
        fs::write(
            output.join("manifest.json"),
            serde_json::to_vec_pretty(manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn existing_archive_retry_rejects_omitted_source_tensor() {
        let directory = tempfile::tempdir().unwrap();
        let (checkpoint, output, recipe) =
            export_source_verification_fixture(directory.path(), "omitted");
        let mut manifest = read_archive_manifest(&output);
        let omitted = manifest.matrices.pop().unwrap();
        fs::remove_file(output.join(omitted.file)).unwrap();
        write_archive_manifest(&output, &manifest);

        assert!(QuantizedArchive::open(&output).is_ok());
        let error = export_safetensors_archive(&checkpoint, &output, &recipe).unwrap_err();
        assert!(
            format!("{error:#}").contains("tensor inventory differs"),
            "{error:#}"
        );
    }

    #[test]
    fn archive_rejects_impossible_packed_size_before_loading_member() {
        let directory = tempfile::tempdir().unwrap();
        let (_, output, _) =
            export_source_verification_fixture(directory.path(), "oversized-member");
        let mut manifest = read_archive_manifest(&output);
        manifest.matrices[0].packed_bytes += 1;
        write_archive_manifest(&output, &manifest);

        let error = QuantizedArchive::open(&output).unwrap_err().to_string();
        assert!(
            error.contains("packed byte count does not match"),
            "{error}"
        );
    }

    #[test]
    fn existing_archive_retry_rejects_reencoded_matrix() {
        let directory = tempfile::tempdir().unwrap();
        let (checkpoint, output, recipe) =
            export_source_verification_fixture(directory.path(), "forged");
        let mut manifest = read_archive_manifest(&output);
        let matrix = &mut manifest.matrices[0];
        let path = output.join(&matrix.file);
        let mut packed = PackedTensor::from_bytes(&fs::read(&path).unwrap()).unwrap();
        packed.codes[0] ^= 1;
        let bytes = packed.to_bytes().unwrap();
        fs::write(&path, &bytes).unwrap();
        matrix.packed_bytes = bytes.len() as u64;
        matrix.sha256 = sha256_identity(&bytes);
        write_archive_manifest(&output, &manifest);

        assert!(QuantizedArchive::open(&output).is_ok());
        let error = export_safetensors_archive(&checkpoint, &output, &recipe).unwrap_err();
        assert!(
            format!("{error:#}").contains("deterministic source encoding"),
            "{error:#}"
        );
    }

    #[test]
    fn existing_archive_retry_rejects_forged_error_metrics() {
        let directory = tempfile::tempdir().unwrap();
        let (checkpoint, output, recipe) =
            export_source_verification_fixture(directory.path(), "forged-errors");
        let mut manifest = read_archive_manifest(&output);
        manifest.matrices[0].mean_squared_error = 0.0;
        manifest.matrices[0].maximum_absolute_error = 0.0;
        write_archive_manifest(&output, &manifest);

        assert!(QuantizedArchive::open(&output).is_ok());
        let error = export_safetensors_archive(&checkpoint, &output, &recipe).unwrap_err();
        assert!(
            format!("{error:#}").contains("error was not derived from source"),
            "{error:#}"
        );
    }

    #[test]
    fn existing_archive_retry_rejects_wrong_source_name_and_shape() {
        let directory = tempfile::tempdir().unwrap();
        let (checkpoint, output, recipe) =
            export_source_verification_fixture(directory.path(), "wrong-name");
        let mut manifest = read_archive_manifest(&output);
        manifest.matrices[0].name = "layers.renamed.weight".to_owned();
        write_archive_manifest(&output, &manifest);
        assert!(QuantizedArchive::open(&output).is_ok());
        assert!(export_safetensors_archive(&checkpoint, &output, &recipe).is_err());

        let (checkpoint, output, recipe) =
            export_source_verification_fixture(directory.path(), "wrong-shape");
        let mut manifest = read_archive_manifest(&output);
        let matrix = &mut manifest.matrices[0];
        let path = output.join(&matrix.file);
        let mut packed = PackedTensor::from_bytes(&fs::read(&path).unwrap()).unwrap();
        packed.shape = vec![packed.elements().unwrap(), 1];
        let bytes = packed.to_bytes().unwrap();
        fs::write(&path, &bytes).unwrap();
        matrix.shape = packed.shape;
        matrix.packed_bytes = bytes.len() as u64;
        matrix.sha256 = sha256_identity(&bytes);
        write_archive_manifest(&output, &manifest);
        assert!(QuantizedArchive::open(&output).is_ok());
        let error = export_safetensors_archive(&checkpoint, &output, &recipe).unwrap_err();
        assert!(
            format!("{error:#}").contains("shape or element count differs"),
            "{error:#}"
        );
    }

    fn alter_first_packed_member(output: &Path) {
        let mut manifest = read_archive_manifest(output);
        let matrix = &mut manifest.matrices[0];
        let path = output.join(&matrix.file);
        let mut packed = PackedTensor::from_bytes(&fs::read(&path).unwrap()).unwrap();
        packed.codes[0] ^= 1;
        let bytes = packed.to_bytes().unwrap();
        fs::write(&path, &bytes).unwrap();
        matrix.packed_bytes = bytes.len() as u64;
        matrix.sha256 = sha256_identity(&bytes);
        write_archive_manifest(output, &manifest);
    }

    #[test]
    fn addressed_open_rejects_archive_swap_after_manifest_capture() {
        let directory = tempfile::tempdir().unwrap();
        let (_, published, _) = export_source_verification_fixture(directory.path(), "published");
        let (_, replacement, _) =
            export_source_verification_fixture(directory.path(), "replacement");
        alter_first_packed_member(&replacement);
        let expected = sha256_identity(&fs::read(published.join("manifest.json")).unwrap());
        let parked = directory.path().join("parked.hquant");

        let error = QuantizedArchive::open_inner_with_hook(&published, Some(&expected), || {
            fs::rename(&published, &parked)?;
            fs::rename(&replacement, &published)?;
            Ok(())
        })
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("SHA-256 mismatch"),
            "{error:#}"
        );
    }

    #[test]
    fn addressed_archive_load_rejects_member_root_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let (_, published, _) =
            export_source_verification_fixture(directory.path(), "published-load");
        let (_, replacement, _) =
            export_source_verification_fixture(directory.path(), "replacement-load");
        alter_first_packed_member(&replacement);
        let expected = sha256_identity(&fs::read(published.join("manifest.json")).unwrap());
        let archive = QuantizedArchive::open_addressed(&published, &expected).unwrap();
        let matrix_name = archive.manifest().matrices[0].name.clone();
        let parked = directory.path().join("parked-load.hquant");
        fs::rename(&published, parked).unwrap();
        fs::rename(&replacement, &published).unwrap();

        let error = archive.load_matrix(&matrix_name).unwrap_err();
        assert!(
            format!("{error:#}").contains("SHA-256 mismatch"),
            "{error:#}"
        );
    }

    #[test]
    fn safetensors_export_is_complete_and_immutable() {
        use safetensors::tensor::TensorView;

        let matrix = (0..137)
            .map(|index| (index as f32 * 0.1).sin())
            .collect::<Vec<_>>();
        let matrix_bytes = matrix
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let norm_bytes = [1.0f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let scalar_bytes = 3.0f32.to_le_bytes();
        let empty_bytes = Vec::<u8>::new();
        let views = vec![
            (
                "layers.0.weight",
                TensorView::new(Dtype::F32, vec![1, 137], &matrix_bytes).unwrap(),
            ),
            (
                "layers.0.norm",
                TensorView::new(Dtype::F32, vec![2], &norm_bytes).unwrap(),
            ),
            (
                "temperature",
                TensorView::new(Dtype::F32, vec![], &scalar_bytes).unwrap(),
            ),
            (
                "empty.buffer",
                TensorView::new(Dtype::F32, vec![0, 2], &empty_bytes).unwrap(),
            ),
        ];
        let checkpoint = safetensors::serialize(views, None).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let checkpoint_path = directory.path().join("weights.safetensors");
        let output = directory.path().join("weights.hquant");
        fs::write(&checkpoint_path, checkpoint).unwrap();
        let recipe = QuantizationRecipe {
            format: UltraQuantFormat::BinaryG128,
            group_size: BONSAI_GROUP_SIZE,
            fake_quant_start_step: 0,
            ternary_warmup_steps: 0,
            distillation_weight: 1.0,
            quantize_embeddings: true,
            quantize_lm_head: true,
        };
        let manifest = export_safetensors_archive(&checkpoint_path, &output, &recipe).unwrap();
        assert_eq!(manifest.matrices.len(), 1);
        assert_eq!(manifest.floating_tensors.len(), 3);
        assert!(output.join("manifest.json").is_file());
        assert!(output.join(&manifest.matrices[0].file).is_file());
        assert!(output.join(&manifest.floating_tensors[0].file).is_file());
        assert_eq!(
            PackedTensor::load(&output.join(&manifest.matrices[0].file))
                .unwrap()
                .decode()
                .unwrap()
                .len(),
            137
        );
        assert_eq!(
            export_safetensors_archive(&checkpoint_path, &output, &recipe).unwrap(),
            manifest
        );

        let second_output = directory.path().join("weights-second.hquant");
        let second_manifest =
            export_safetensors_archive(&checkpoint_path, &second_output, &recipe).unwrap();
        assert_eq!(second_manifest, manifest);
        assert_eq!(
            QuantizedArchive::open(&second_output)
                .unwrap()
                .content_hash()
                .unwrap(),
            QuantizedArchive::open(&output)
                .unwrap()
                .content_hash()
                .unwrap()
        );

        let archive = QuantizedArchive::open(&output).unwrap();
        archive.verify_source_checkpoint(&checkpoint_path).unwrap();
        #[cfg(unix)]
        {
            let checkpoint_link = directory.path().join("weights-link.safetensors");
            std::os::unix::fs::symlink(&checkpoint_path, &checkpoint_link).unwrap();
            assert!(archive.verify_source_checkpoint(&checkpoint_link).is_err());
        }
        let other_checkpoint = directory.path().join("other.safetensors");
        fs::write(&other_checkpoint, b"different checkpoint").unwrap();
        assert!(archive.verify_source_checkpoint(&other_checkpoint).is_err());
        assert_eq!(
            archive.load_matrix("layers.0.weight").unwrap().shape,
            vec![1, 137]
        );
        assert_eq!(
            archive.load_floating("layers.0.norm").unwrap().bytes,
            norm_bytes
        );

        let mut backend = MockArchiveBackend::default();
        archive.load_into(&mut backend).unwrap();
        assert!(backend.committed);
        assert_eq!(backend.matrices, vec!["layers.0.weight"]);
        assert_eq!(backend.floating.len(), 3);
        assert!(backend.floating.contains(&"layers.0.norm".to_owned()));
        assert!(backend.floating.contains(&"temperature".to_owned()));
        assert!(backend.floating.contains(&"empty.buffer".to_owned()));

        let mut failing = MockArchiveBackend {
            fail_floating: true,
            ..Default::default()
        };
        assert!(archive.load_into(&mut failing).is_err());
        assert!(failing.rolled_back);
        assert!(!failing.committed);

        let manifest_path = output.join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path).unwrap();
        let mut unsafe_manifest: serde_json::Value =
            serde_json::from_slice(&manifest_bytes).unwrap();
        unsafe_manifest["matrices"][0]["file"] = serde_json::json!("../escape.hquant");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&unsafe_manifest).unwrap(),
        )
        .unwrap();
        assert!(QuantizedArchive::open(&output).is_err());
        fs::write(&manifest_path, manifest_bytes).unwrap();

        let unexpected = output.join("unverified-payload.bin");
        fs::write(&unexpected, b"not in manifest").unwrap();
        assert!(QuantizedArchive::open(&output).is_err());
        fs::remove_file(unexpected).unwrap();

        let norm_manifest = manifest
            .floating_tensors
            .iter()
            .find(|tensor| tensor.name == "layers.0.norm")
            .unwrap();
        let floating_path = output.join(&norm_manifest.file);
        let mut corrupt = fs::read(&floating_path).unwrap();
        corrupt[0] ^= 1;
        fs::write(&floating_path, corrupt).unwrap();
        assert!(QuantizedArchive::open(&output).is_err());
    }

    #[derive(Default)]
    struct MockArchiveBackend {
        matrices: Vec<String>,
        floating: Vec<String>,
        committed: bool,
        rolled_back: bool,
        fail_floating: bool,
    }

    impl QuantizedModelBackend for MockArchiveBackend {
        fn begin_archive_load(&mut self, _: &QuantizationManifest) -> Result<()> {
            Ok(())
        }

        fn load_quantized_matrix(&mut self, name: &str, _: &PackedTensor) -> Result<()> {
            self.matrices.push(name.to_owned());
            Ok(())
        }

        fn load_floating_tensor(&mut self, tensor: &FloatingTensorData) -> Result<()> {
            if self.fail_floating {
                bail!("injected floating-tensor load failure");
            }
            self.floating.push(tensor.name.clone());
            Ok(())
        }

        fn commit_archive_load(&mut self) -> Result<()> {
            self.committed = true;
            Ok(())
        }

        fn rollback_archive_load(&mut self) -> Result<()> {
            self.rolled_back = true;
            self.matrices.clear();
            self.floating.clear();
            Ok(())
        }
    }
}
