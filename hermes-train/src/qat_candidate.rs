//! Immutable publication of a trained QAT model and its HQUANT archive.
//!
//! A stable candidate key is a write-once identity. Publication first creates
//! a canonical SafeTensors snapshot and complete HQUANT archive in a private
//! sibling directory, validates every archive member, and then exposes the
//! whole candidate with one directory rename. Retrying a completed operation
//! returns the existing artifact only when the model bytes and recipe match.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use burn::module::AutodiffModule;
use hermes_llm::{Transformer, save_safetensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::quantization::{
    QuantizationManifest, QuantizationRecipe, QuantizedArchive, export_safetensors_archive,
};

const CANDIDATE_VERSION: u32 = 1;
const CANDIDATE_MANIFEST: &str = "candidate.json";
const WEIGHTS_FILE: &str = "weights.safetensors";
const ARCHIVE_DIRECTORY: &str = "hquant";
const ARCHIVE_MANIFEST: &str = "manifest.json";
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Exact aggregate measurements derived from the validated archive manifest.
///
/// `packed_bytes` covers HQUANT matrices (including their group scales).
/// `archive_weight_bytes` additionally includes tensors deliberately retained
/// in their source dtype. Consequently `average_bits_per_weight` is the true
/// average across every stored model element, not the nominal codec rate.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QatArchiveMetrics {
    pub quantized_tensors: u64,
    pub quantized_elements: u64,
    pub packed_bytes: u64,
    pub floating_tensors: u64,
    pub floating_elements: u64,
    pub floating_bytes: u64,
    pub archive_weight_elements: u64,
    pub archive_weight_bytes: u64,
    pub average_bits_per_weight: f64,
    pub weighted_mean_squared_error: f64,
    pub maximum_absolute_error: f64,
}

/// Write-once receipt sealed alongside one QAT candidate.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QatCandidateManifest {
    pub version: u32,
    pub candidate_key: String,
    pub weights_file: String,
    pub weights_bytes: u64,
    pub weights_sha256: String,
    pub archive_directory: String,
    pub archive_manifest: String,
    pub archive_manifest_sha256: String,
    pub archive_content_sha256: String,
    pub recipe: QuantizationRecipe,
    pub metrics: QatArchiveMetrics,
}

/// Paths, content identities, and measurements returned after final-path
/// validation. Paths point only at the sealed candidate, never at staging.
#[derive(Clone, Debug, PartialEq)]
pub struct QatCandidatePublication {
    pub candidate_key: String,
    pub candidate_root: PathBuf,
    pub candidate_manifest_path: PathBuf,
    pub candidate_manifest_sha256: String,
    pub weights_path: PathBuf,
    pub weights_sha256: String,
    pub archive_path: PathBuf,
    pub archive_manifest_path: PathBuf,
    pub archive_manifest_sha256: String,
    pub archive_content_sha256: String,
    pub metrics: QatArchiveMetrics,
}

/// Snapshot a trained model and publish its HQUANT candidate under `root/key`.
///
/// The key must be a single conservative path component. Once published, a
/// key cannot be rebound: a retry with different canonical model bytes or a
/// different recipe fails. Existing artifacts are reopened and fully checked
/// before they are returned.
pub fn publish_qat_candidate(
    model: &Transformer,
    root: &Path,
    key: &str,
    recipe: &QuantizationRecipe,
) -> Result<QatCandidatePublication> {
    publish_qat_candidate_with_hook(model, root, key, recipe, |_, _| Ok(()))
}

/// Reopen an already published candidate and authenticate its receipt,
/// canonical weights, HQUANT manifest, and every referenced archive member.
pub fn open_qat_candidate(candidate_root: &Path) -> Result<QatCandidatePublication> {
    let key = candidate_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("QAT candidate key is not UTF-8")?;
    validate_candidate_key(key)?;
    validate_candidate(candidate_root, key)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublicationPoint {
    WeightsSynced,
    ArchiveVerified,
    ManifestSynced,
}

fn publish_qat_candidate_with_hook<F>(
    model: &Transformer,
    root: &Path,
    key: &str,
    recipe: &QuantizationRecipe,
    mut hook: F,
) -> Result<QatCandidatePublication>
where
    F: FnMut(PublicationPoint, &mut StagingDirectory) -> Result<()>,
{
    validate_candidate_key(key)?;
    recipe.validate()?;
    ensure_directory(root, "QAT candidate root")?;

    let candidate_root = root.join(key);
    if let Ok(metadata) = fs::symlink_metadata(&candidate_root) {
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "existing QAT candidate must be a real directory"
        );
    }

    let staging_path = allocate_staging_directory(root, key)?;
    let mut staging = StagingDirectory::new(staging_path);
    let staged_weights = staging.path().join(WEIGHTS_FILE);
    save_safetensors(&model.clone().valid(), &staged_weights)
        .context("failed to snapshot trained QAT weights")?;
    OpenOptions::new()
        .read(true)
        .open(&staged_weights)?
        .sync_all()?;
    let (weights_bytes, weights_sha256) = hash_regular_file(&staged_weights, "QAT weights")?;
    hook(PublicationPoint::WeightsSynced, &mut staging)?;

    if candidate_root.exists() {
        let publication = validate_candidate(&candidate_root, key)?;
        ensure!(
            publication.weights_sha256 == weights_sha256,
            "QAT candidate key '{key}' is already bound to different model weights"
        );
        let existing: QatCandidateManifest = read_json_regular(
            &publication.candidate_manifest_path,
            "QAT candidate manifest",
        )?;
        ensure!(
            existing.recipe == *recipe,
            "QAT candidate key '{key}' is already bound to a different quantization recipe"
        );
        return Ok(publication);
    }

    let staged_archive = staging.path().join(ARCHIVE_DIRECTORY);
    export_safetensors_archive(&staged_weights, &staged_archive, recipe)
        .context("failed to export trained QAT weights as HQUANT")?;
    let archive = QuantizedArchive::open(&staged_archive)
        .context("failed to reopen staged HQUANT archive")?;
    archive
        .verify_source_checkpoint(&staged_weights)
        .context("staged HQUANT archive is not bound to its canonical weights")?;
    ensure!(
        archive.manifest().recipe == *recipe,
        "staged HQUANT archive changed the requested recipe"
    );
    let archive_content_sha256 = archive.content_hash()?;
    let archive_manifest_path = staged_archive.join(ARCHIVE_MANIFEST);
    let (_, archive_manifest_sha256) =
        hash_regular_file(&archive_manifest_path, "HQUANT manifest")?;
    let metrics = aggregate_metrics(archive.manifest())?;
    hook(PublicationPoint::ArchiveVerified, &mut staging)?;

    let manifest = QatCandidateManifest {
        version: CANDIDATE_VERSION,
        candidate_key: key.to_owned(),
        weights_file: WEIGHTS_FILE.to_owned(),
        weights_bytes,
        weights_sha256: weights_sha256.clone(),
        archive_directory: ARCHIVE_DIRECTORY.to_owned(),
        archive_manifest: format!("{ARCHIVE_DIRECTORY}/{ARCHIVE_MANIFEST}"),
        archive_manifest_sha256,
        archive_content_sha256,
        recipe: recipe.clone(),
        metrics,
    };
    write_new_synced(
        &staging.path().join(CANDIDATE_MANIFEST),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    sync_directory(staging.path())?;
    hook(PublicationPoint::ManifestSynced, &mut staging)?;

    // Validate through the same public-facing path logic before publication.
    validate_candidate(staging.path(), key)?;
    match fs::rename(staging.path(), &candidate_root) {
        Ok(()) => staging.disarm(),
        Err(_error) if candidate_root.exists() => {
            // A concurrent writer won the key. It is acceptable only when it
            // published byte-for-byte the same model and recipe.
            let publication = validate_candidate(&candidate_root, key)?;
            ensure!(
                publication.weights_sha256 == weights_sha256,
                "concurrent QAT publication bound key '{key}' to different model weights"
            );
            let existing: QatCandidateManifest = read_json_regular(
                &publication.candidate_manifest_path,
                "QAT candidate manifest",
            )?;
            ensure!(
                existing.recipe == *recipe,
                "concurrent QAT publication bound key '{key}' to a different recipe"
            );
            return Ok(publication);
        }
        Err(error) => {
            return Err(error).with_context(|| format!("failed to publish QAT candidate '{key}'"));
        }
    }
    sync_directory(root)?;

    let publication = validate_candidate(&candidate_root, key)?;
    ensure!(
        publication.weights_sha256 == weights_sha256,
        "published QAT candidate weights changed during final validation"
    );
    Ok(publication)
}

fn aggregate_metrics(manifest: &QuantizationManifest) -> Result<QatArchiveMetrics> {
    let quantized_tensors =
        u64::try_from(manifest.matrices.len()).context("quantized tensor count exceeds u64")?;
    let quantized_elements = checked_sum(
        manifest.matrices.iter().map(|matrix| matrix.elements),
        "quantized element count",
    )?;
    ensure!(
        quantized_tensors > 0 && quantized_elements > 0,
        "HQUANT archive contains no quantized weights"
    );
    let packed_bytes = checked_sum(
        manifest.matrices.iter().map(|matrix| matrix.packed_bytes),
        "packed byte count",
    )?;
    let floating_tensors = u64::try_from(manifest.floating_tensors.len())
        .context("floating tensor count exceeds u64")?;
    let floating_elements = checked_sum(
        manifest
            .floating_tensors
            .iter()
            .map(|tensor| tensor.elements),
        "floating element count",
    )?;
    let floating_bytes = checked_sum(
        manifest.floating_tensors.iter().map(|tensor| tensor.bytes),
        "floating byte count",
    )?;
    let archive_weight_elements = quantized_elements
        .checked_add(floating_elements)
        .context("archive weight element count overflows u64")?;
    let archive_weight_bytes = packed_bytes
        .checked_add(floating_bytes)
        .context("archive weight byte count overflows u64")?;

    // Compensated summation makes the weighted aggregate stable even for
    // archives containing many matrices with very different sizes.
    let mut weighted_squared_error = 0.0f64;
    let mut compensation = 0.0f64;
    let mut maximum_absolute_error = 0.0f64;
    for matrix in &manifest.matrices {
        ensure!(
            matrix.mean_squared_error.is_finite() && matrix.mean_squared_error >= 0.0,
            "matrix '{}' has invalid quantization error",
            matrix.name
        );
        ensure!(
            matrix.maximum_absolute_error.is_finite() && matrix.maximum_absolute_error >= 0.0,
            "matrix '{}' has invalid maximum quantization error",
            matrix.name
        );
        let contribution = matrix.mean_squared_error * matrix.elements as f64;
        ensure!(
            contribution.is_finite(),
            "matrix '{}' weighted quantization error overflows",
            matrix.name
        );
        let corrected = contribution - compensation;
        let next = weighted_squared_error + corrected;
        compensation = (next - weighted_squared_error) - corrected;
        weighted_squared_error = next;
        maximum_absolute_error =
            maximum_absolute_error.max(f64::from(matrix.maximum_absolute_error));
    }
    let weighted_mean_squared_error = weighted_squared_error / quantized_elements as f64;
    ensure!(
        weighted_mean_squared_error.is_finite(),
        "aggregate weighted quantization error is not finite"
    );
    let average_bits_per_weight = manifest.true_average_bits_per_weight()?;

    Ok(QatArchiveMetrics {
        quantized_tensors,
        quantized_elements,
        packed_bytes,
        floating_tensors,
        floating_elements,
        floating_bytes,
        archive_weight_elements,
        archive_weight_bytes,
        average_bits_per_weight,
        weighted_mean_squared_error,
        maximum_absolute_error,
    })
}

fn validate_candidate(root: &Path, expected_key: &str) -> Result<QatCandidatePublication> {
    ensure_real_directory(root, "QAT candidate")?;
    validate_candidate_inventory(root)?;
    let candidate_manifest_path = root.join(CANDIDATE_MANIFEST);
    let manifest_bytes = read_regular(&candidate_manifest_path, "QAT candidate manifest")?;
    let candidate_manifest_sha256 = sha256_label(&manifest_bytes);
    let manifest: QatCandidateManifest =
        serde_json::from_slice(&manifest_bytes).context("invalid QAT candidate manifest")?;
    ensure!(
        manifest.version == CANDIDATE_VERSION,
        "unsupported QAT candidate version {}",
        manifest.version
    );
    validate_candidate_key(&manifest.candidate_key)?;
    ensure!(
        manifest.candidate_key == expected_key,
        "QAT candidate manifest key does not match its requested identity"
    );
    ensure!(
        manifest.weights_file == WEIGHTS_FILE
            && manifest.archive_directory == ARCHIVE_DIRECTORY
            && manifest.archive_manifest == format!("{ARCHIVE_DIRECTORY}/{ARCHIVE_MANIFEST}"),
        "QAT candidate uses a non-canonical artifact layout"
    );
    manifest.recipe.validate()?;

    let weights_path = root.join(WEIGHTS_FILE);
    let (weights_bytes, weights_sha256) = hash_regular_file(&weights_path, "QAT weights")?;
    ensure!(
        weights_bytes == manifest.weights_bytes && weights_sha256 == manifest.weights_sha256,
        "QAT candidate weights do not match their manifest"
    );

    let archive_path = root.join(ARCHIVE_DIRECTORY);
    let archive = QuantizedArchive::open(&archive_path)
        .context("QAT candidate contains an invalid HQUANT archive")?;
    archive.verify_source_checkpoint(&weights_path)?;
    ensure!(
        archive.manifest().recipe == manifest.recipe,
        "QAT candidate archive recipe does not match its manifest"
    );
    let archive_manifest_path = archive_path.join(ARCHIVE_MANIFEST);
    let (_, archive_manifest_sha256) =
        hash_regular_file(&archive_manifest_path, "HQUANT manifest")?;
    let archive_content_sha256 = archive.content_hash()?;
    ensure!(
        archive_manifest_sha256 == manifest.archive_manifest_sha256
            && archive_content_sha256 == manifest.archive_content_sha256,
        "QAT candidate archive identity does not match its manifest"
    );
    let metrics = aggregate_metrics(archive.manifest())?;
    ensure!(
        metrics == manifest.metrics,
        "QAT candidate aggregate metrics do not match its archive"
    );

    Ok(QatCandidatePublication {
        candidate_key: manifest.candidate_key,
        candidate_root: root.to_path_buf(),
        candidate_manifest_path,
        candidate_manifest_sha256,
        weights_path,
        weights_sha256,
        archive_path,
        archive_manifest_path,
        archive_manifest_sha256,
        archive_content_sha256,
        metrics,
    })
}

fn validate_candidate_inventory(root: &Path) -> Result<()> {
    let expected = BTreeSet::from([
        CANDIDATE_MANIFEST.to_owned(),
        WEIGHTS_FILE.to_owned(),
        ARCHIVE_DIRECTORY.to_owned(),
    ]);
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("QAT candidate contains a non-UTF-8 entry"))?;
        actual.insert(name);
    }
    ensure!(
        actual == expected,
        "QAT candidate contains missing or unverified top-level entries"
    );
    let archive = fs::symlink_metadata(root.join(ARCHIVE_DIRECTORY))?;
    ensure!(
        archive.is_dir() && !archive.file_type().is_symlink(),
        "QAT archive must be a real directory"
    );
    Ok(())
}

fn validate_candidate_key(key: &str) -> Result<()> {
    ensure!(
        !key.is_empty() && key.len() <= 128,
        "QAT candidate key must contain 1..=128 bytes"
    );
    let path = Path::new(key);
    let mut components = path.components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "QAT candidate key must be one normalized path component"
    );
    ensure!(
        key.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            && key != "."
            && key != "..",
        "QAT candidate key may contain only ASCII letters, digits, '.', '-', and '_'"
    );
    Ok(())
}

fn ensure_directory(path: &Path, label: &str) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "{label} must be a real directory: {}",
                path.display()
            );
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path)
                .with_context(|| format!("failed to create {label} {}", path.display()))?;
            ensure_real_directory(path, label)
        }
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

fn ensure_real_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} must be a real directory: {}",
        path.display()
    );
    Ok(())
}

fn allocate_staging_directory(root: &Path, key: &str) -> Result<PathBuf> {
    for _ in 0..1024 {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(".{key}.qat-tmp-{}-{sequence}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create QAT staging directory {}", path.display())
                });
            }
        }
    }
    bail!("could not allocate a QAT candidate staging directory")
}

fn checked_sum(mut values: impl Iterator<Item = u64>, label: &str) -> Result<u64> {
    values.try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .with_context(|| format!("{label} overflows u64"))
    })
}

fn read_json_regular<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Result<T> {
    serde_json::from_slice(&read_regular(path, label)?)
        .with_context(|| format!("invalid {label} {}", path.display()))
}

fn read_regular(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} must be a regular file: {}",
        path.display()
    );
    fs::read(path).with_context(|| format!("failed to read {label} {}", path.display()))
}

fn hash_regular_file(path: &Path, label: &str) -> Result<(u64, String)> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} must be a regular file: {}",
        path.display()
    );
    let mut input = File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes = bytes
            .checked_add(u64::try_from(read).context("file chunk length exceeds u64")?)
            .context("artifact byte count overflows u64")?;
    }
    Ok((bytes, format!("sha256:{:x}", digest.finalize())))
}

fn sha256_label(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to create immutable file {}", path.display()))?;
    output.write_all(bytes)?;
    output.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

struct StagingDirectory {
    path: PathBuf,
    armed: bool,
}

impl StagingDirectory {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(test)]
    fn preserve_for_crash_test(&mut self) {
        self.armed = false;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if self.armed && self.path.exists() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    use crate::quantization::{BONSAI_GROUP_SIZE, UltraQuantFormat};

    fn model(vocab_size: usize) -> Transformer {
        let device = Device::default().autodiff();
        let mut config = hermes_llm::get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = vocab_size;
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
        Transformer::new(&config, &device).unwrap()
    }

    fn recipe(format: UltraQuantFormat) -> QuantizationRecipe {
        QuantizationRecipe {
            format,
            group_size: BONSAI_GROUP_SIZE,
            fake_quant_start_step: 0,
            ternary_warmup_steps: 0,
            distillation_weight: 0.0,
            quantize_embeddings: true,
            quantize_lm_head: true,
        }
    }

    #[test]
    fn publishes_validated_candidate_with_exact_metrics_and_replays_idempotently() {
        let directory = tempfile::tempdir().unwrap();
        let model = model(32);
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        let first = publish_qat_candidate(&model, directory.path(), "step-42", &recipe).unwrap();
        let second = publish_qat_candidate(&model, directory.path(), "step-42", &recipe).unwrap();
        assert_eq!(first, second);
        assert!(first.metrics.quantized_tensors > 0);
        assert!(first.metrics.quantized_elements > 0);
        assert!(first.metrics.packed_bytes > 0);
        assert!(first.metrics.average_bits_per_weight > 0.0);
        assert!(first.metrics.weighted_mean_squared_error >= 0.0);
        assert!(first.metrics.maximum_absolute_error >= 0.0);
        assert_eq!(
            hash_regular_file(&first.candidate_manifest_path, "candidate")
                .unwrap()
                .1,
            first.candidate_manifest_sha256
        );
        assert_eq!(
            hash_regular_file(&first.archive_manifest_path, "archive manifest")
                .unwrap()
                .1,
            first.archive_manifest_sha256
        );
        let archive = QuantizedArchive::open(&first.archive_path).unwrap();
        archive
            .verify_source_checkpoint(&first.weights_path)
            .unwrap();
        assert_eq!(
            archive.content_hash().unwrap(),
            first.archive_content_sha256
        );
        assert_eq!(
            first.metrics.archive_weight_bytes,
            first.metrics.packed_bytes + first.metrics.floating_bytes
        );
        assert_eq!(
            first.metrics.archive_weight_elements,
            first.metrics.quantized_elements + first.metrics.floating_elements
        );
    }

    #[test]
    fn stable_key_rejects_different_weights_and_recipe() {
        let directory = tempfile::tempdir().unwrap();
        let first_model = model(32);
        let binary = recipe(UltraQuantFormat::BinaryG128);
        publish_qat_candidate(&first_model, directory.path(), "stable", &binary).unwrap();

        let different_model = model(33);
        let weights_error =
            publish_qat_candidate(&different_model, directory.path(), "stable", &binary)
                .unwrap_err();
        assert!(
            weights_error
                .to_string()
                .contains("different model weights")
        );

        let ternary = recipe(UltraQuantFormat::TernaryG128);
        let recipe_error =
            publish_qat_candidate(&first_model, directory.path(), "stable", &ternary).unwrap_err();
        assert!(
            recipe_error
                .to_string()
                .contains("different quantization recipe")
        );
    }

    #[test]
    fn simulated_crashes_at_each_durable_boundary_are_retryable() {
        let directory = tempfile::tempdir().unwrap();
        let model = model(32);
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        for (index, fault) in [
            PublicationPoint::WeightsSynced,
            PublicationPoint::ArchiveVerified,
            PublicationPoint::ManifestSynced,
        ]
        .into_iter()
        .enumerate()
        {
            let key = format!("crash-{index}");
            let error = publish_qat_candidate_with_hook(
                &model,
                directory.path(),
                &key,
                &recipe,
                |point, staging| {
                    if point == fault {
                        // Leaving the staging directory emulates abrupt process
                        // death, for which Drop cleanup never runs.
                        staging.preserve_for_crash_test();
                        bail!("simulated crash at {point:?}");
                    }
                    Ok(())
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains("simulated crash"));
            assert!(!directory.path().join(&key).exists());
            assert!(
                fs::read_dir(directory.path())
                    .unwrap()
                    .filter_map(Result::ok)
                    .any(|entry| entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(&format!(".{key}.qat-tmp-")))
            );
            let publication =
                publish_qat_candidate(&model, directory.path(), &key, &recipe).unwrap();
            assert!(publication.candidate_root.is_dir());
            QuantizedArchive::open(&publication.archive_path).unwrap();
        }
    }

    #[test]
    fn tampering_is_rejected_without_replacing_the_candidate() {
        let directory = tempfile::tempdir().unwrap();
        let model = model(32);
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        let publication =
            publish_qat_candidate(&model, directory.path(), "tamper", &recipe).unwrap();
        let archive = QuantizedArchive::open(&publication.archive_path).unwrap();
        let member = publication
            .archive_path
            .join(&archive.manifest().matrices[0].file);
        let mut corrupt = fs::read(&member).unwrap();
        corrupt[0] ^= 1;
        fs::write(&member, corrupt).unwrap();

        let error = publish_qat_candidate(&model, directory.path(), "tamper", &recipe).unwrap_err();
        assert!(error.to_string().contains("invalid HQUANT archive"));
        assert!(publication.candidate_root.exists());
    }

    #[test]
    fn unsafe_keys_and_top_level_extras_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let model = model(32);
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        for key in ["", "../escape", "nested/key", ".", "bad key"] {
            assert!(publish_qat_candidate(&model, directory.path(), key, &recipe).is_err());
        }
        let publication =
            publish_qat_candidate(&model, directory.path(), "closed", &recipe).unwrap();
        fs::write(publication.candidate_root.join("extra"), b"unverified").unwrap();
        let error = publish_qat_candidate(&model, directory.path(), "closed", &recipe).unwrap_err();
        assert!(error.to_string().contains("unverified top-level"));
    }
}
