//! Immutable publication of a trained QAT model and its HQUANT archive.
//!
//! A stable candidate key is a write-once identity. Publication first creates
//! a canonical SafeTensors snapshot and complete HQUANT archive in a private
//! sibling directory, validates every archive member, and then exposes the
//! whole candidate with one directory rename. Retrying a completed operation
//! returns the existing artifact only when the model bytes and recipe match.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use burn::module::AutodiffModule;
use serde::{Deserialize, Serialize};
use summa_llm::{Transformer, save_safetensors};

use crate::artifact_io::{
    ensure_directory, ensure_real_directory, hash_regular_file, read_regular_bounded,
    sha256_identity as sha256_label, sync_directory, sync_regular_file, validate_sha256_identity,
    write_new_synced,
};

use crate::quantization::{
    QuantizationManifest, QuantizationRecipe, QuantizedArchive, export_safetensors_archive,
};

const CANDIDATE_VERSION: u32 = 1;
const CANDIDATE_MANIFEST: &str = "candidate.json";
const WEIGHTS_FILE: &str = "weights.safetensors";
const ARCHIVE_DIRECTORY: &str = "hquant";
const ARCHIVE_MANIFEST: &str = "manifest.json";
const MAX_QAT_JSON_BYTES: u64 = 16 * 1024 * 1024;
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
    pub recipe: QuantizationRecipe,
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
    publish_qat_candidate_with_hook(model, root, key, recipe, None, |_, _| Ok(()))
}

/// Publish or reopen a candidate bound to an independently authenticated
/// canonical model snapshot.
///
/// When `root/key` is already sealed, `source_weights_sha256` avoids
/// serializing `model` again. Because a crash may leave an orphan candidate
/// whose manifest has not yet been sealed into trainer state, adoption still
/// reproduces that archive once from its canonical weights. Callers must
/// obtain the digest from immutable provenance such as a verified trainer
/// checkpoint or a content-addressed sleep checkpoint; mutable in-memory state
/// is not a valid source of this identity.
pub fn publish_qat_candidate_from_authenticated_source(
    model: &Transformer,
    root: &Path,
    key: &str,
    recipe: &QuantizationRecipe,
    source_weights_sha256: &str,
) -> Result<QatCandidatePublication> {
    validate_sha256_identity(source_weights_sha256, "QAT source weights identity")?;
    publish_qat_candidate_with_hook(
        model,
        root,
        key,
        recipe,
        Some(source_weights_sha256),
        |_, _| Ok(()),
    )
}

/// Reopen an already published candidate and authenticate its receipt,
/// canonical weights, HQUANT manifest, and every referenced archive member.
pub fn open_qat_candidate(candidate_root: &Path) -> Result<QatCandidatePublication> {
    let key = candidate_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("QAT candidate key is not UTF-8")?;
    validate_candidate_key(key)?;
    // Without an independently sealed receipt, internally consistent hashes
    // are not provenance: a forged archive can rehash its own wrong codes and
    // claim the canonical source digest. Reproduce the deterministic encoding
    // here; checkpoint/benchmark resume uses the addressed fast path below.
    validate_candidate_reproduced(candidate_root, key)
}

/// Reopen a candidate whose exact receipt bytes are sealed by an immutable
/// trainer checkpoint or benchmark manifest.
///
/// The small receipt is authenticated before any model weights or archive
/// members are read, avoiding expensive validation of a candidate that cannot
/// match the caller's durable identity.
pub fn open_qat_candidate_addressed(
    candidate_root: &Path,
    expected_candidate_manifest_sha256: &str,
) -> Result<QatCandidatePublication> {
    validate_sha256_identity(
        expected_candidate_manifest_sha256,
        "QAT candidate manifest identity",
    )?;
    let key = candidate_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("QAT candidate key is not UTF-8")?;
    validate_candidate_key(key)?;
    validate_candidate_inner_with_hook(
        candidate_root,
        key,
        false,
        Some(expected_candidate_manifest_sha256),
        || Ok(()),
    )
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
    source_weights_sha256: Option<&str>,
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

    // A sealed source digest is sufficient to bind an existing candidate to
    // the trainer checkpoint. Authenticate the candidate first; only a new
    // publication needs to materialize canonical SafeTensors bytes from the
    // accelerator-backed model.
    if let Some(expected) = source_weights_sha256
        && candidate_root.exists()
    {
        let publication = validate_candidate_reproduced(&candidate_root, key)?;
        ensure!(
            publication.weights_sha256 == expected,
            "QAT candidate key '{key}' is bound to different model weights"
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

    let staging_path = allocate_staging_directory(root, key)?;
    let mut staging = StagingDirectory::new(staging_path);
    let staged_weights = staging.path().join(WEIGHTS_FILE);
    save_safetensors(&model.clone().valid(), &staged_weights)
        .context("failed to snapshot trained QAT weights")?;
    sync_regular_file(&staged_weights, "QAT weights")?;
    let (weights_bytes, weights_sha256) = hash_regular_file(&staged_weights)?;
    if let Some(expected) = source_weights_sha256 {
        ensure!(
            weights_sha256 == expected,
            "canonical QAT weights {weights_sha256} differ from authenticated source {expected}"
        );
    }
    hook(PublicationPoint::WeightsSynced, &mut staging)?;

    if candidate_root.exists() {
        let publication = validate_candidate_reproduced(&candidate_root, key)?;
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
    // `export_safetensors_archive` performs the one expensive deterministic
    // derivation check for a newly created archive. Reopening it here only
    // needs to authenticate the exact members and their sealed source digest.
    archive
        .verify_source_identity(&weights_sha256)
        .context("staged HQUANT archive is not bound to its canonical weights")?;
    ensure!(
        archive.manifest().recipe == *recipe,
        "staged HQUANT archive changed the requested recipe"
    );
    let archive_content_sha256 = archive.content_hash()?;
    let archive_manifest_path = staged_archive.join(ARCHIVE_MANIFEST);
    let (_, archive_manifest_sha256) = hash_regular_file(&archive_manifest_path)?;
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
    let candidate_manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let staged_candidate_manifest_sha256 = sha256_label(&candidate_manifest_bytes);
    write_new_synced(
        &staging.path().join(CANDIDATE_MANIFEST),
        &candidate_manifest_bytes,
    )?;
    sync_directory(staging.path())?;
    hook(PublicationPoint::ManifestSynced, &mut staging)?;

    // Validate through the same public-facing path logic before publication.
    let staged_publication = validate_candidate(staging.path(), key)?;
    ensure!(
        staged_publication.candidate_manifest_sha256 == staged_candidate_manifest_sha256,
        "staged QAT candidate manifest changed during validation"
    );
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
            ensure!(
                publication.candidate_manifest_sha256 == staged_candidate_manifest_sha256,
                "concurrent QAT publication bound key '{key}' to different candidate bytes"
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
        publication.weights_sha256 == weights_sha256
            && publication.candidate_manifest_sha256 == staged_candidate_manifest_sha256,
        "published QAT candidate changed during final validation"
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
    validate_candidate_inner(root, expected_key, false)
}

fn validate_candidate_reproduced(
    root: &Path,
    expected_key: &str,
) -> Result<QatCandidatePublication> {
    validate_candidate_inner(root, expected_key, true)
        .context("existing QAT candidate does not reproduce its canonical weights")
}

fn validate_candidate_inner(
    root: &Path,
    expected_key: &str,
    reproduce_source: bool,
) -> Result<QatCandidatePublication> {
    validate_candidate_inner_with_hook(root, expected_key, reproduce_source, None, || Ok(()))
}

fn validate_candidate_inner_with_hook(
    root: &Path,
    expected_key: &str,
    reproduce_source: bool,
    expected_candidate_manifest_sha256: Option<&str>,
    after_manifest: impl FnOnce() -> Result<()>,
) -> Result<QatCandidatePublication> {
    ensure_real_directory(root, "QAT candidate")?;
    validate_candidate_inventory(root)?;
    let candidate_manifest_path = root.join(CANDIDATE_MANIFEST);
    let manifest_bytes = read_regular_bounded(
        &candidate_manifest_path,
        MAX_QAT_JSON_BYTES,
        "QAT candidate manifest",
    )?;
    let candidate_manifest_sha256 = sha256_label(&manifest_bytes);
    if let Some(expected) = expected_candidate_manifest_sha256 {
        ensure!(
            candidate_manifest_sha256 == expected,
            "QAT candidate receipt identity mismatch"
        );
    }
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
    after_manifest()?;

    let weights_path = root.join(WEIGHTS_FILE);
    let (weights_bytes, weights_sha256) = hash_regular_file(&weights_path)?;
    ensure!(
        weights_bytes == manifest.weights_bytes && weights_sha256 == manifest.weights_sha256,
        "QAT candidate weights do not match their manifest"
    );

    let archive_path = root.join(ARCHIVE_DIRECTORY);
    let archive =
        QuantizedArchive::open_addressed(&archive_path, &manifest.archive_manifest_sha256)
            .context("QAT candidate contains an invalid HQUANT archive")?;
    archive.verify_source_identity(&weights_sha256)?;
    if reproduce_source {
        archive.verify_source_checkpoint(&weights_path)?;
    }
    ensure!(
        archive.manifest().recipe == manifest.recipe,
        "QAT candidate archive recipe does not match its manifest"
    );
    let archive_manifest_path = archive_path.join(ARCHIVE_MANIFEST);
    let archive_manifest_sha256 = archive.manifest_sha256().to_owned();
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
        recipe: manifest.recipe,
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
        ensure!(
            expected.contains(&name) && actual.insert(name),
            "QAT candidate contains missing or unverified top-level entries"
        );
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
    serde_json::from_slice(&read_regular_bounded(path, MAX_QAT_JSON_BYTES, label)?)
        .with_context(|| format!("invalid {label} {}", path.display()))
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
        let mut config = summa_llm::get_builtin_model("hybrid-tiny").unwrap();
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

    fn copy_tree(source: &Path, destination: &Path) {
        fs::create_dir(destination).unwrap();
        for entry in fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let target = destination.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
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
        assert_eq!(
            open_qat_candidate_addressed(&first.candidate_root, &first.candidate_manifest_sha256,)
                .unwrap(),
            first
        );
        let wrong_receipt = format!("sha256:{}", "0".repeat(64));
        let error = open_qat_candidate_addressed(&first.candidate_root, &wrong_receipt)
            .unwrap_err()
            .to_string();
        assert!(error.contains("receipt identity mismatch"), "{error}");
        assert!(first.metrics.quantized_tensors > 0);
        assert!(first.metrics.quantized_elements > 0);
        assert!(first.metrics.packed_bytes > 0);
        assert!(first.metrics.average_bits_per_weight > 0.0);
        assert!(first.metrics.weighted_mean_squared_error >= 0.0);
        assert!(first.metrics.maximum_absolute_error >= 0.0);
        assert_eq!(
            hash_regular_file(&first.candidate_manifest_path).unwrap().1,
            first.candidate_manifest_sha256
        );
        assert_eq!(
            hash_regular_file(&first.archive_manifest_path).unwrap().1,
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
    fn authenticated_source_retry_reopens_without_serializing_mutable_model_state() {
        let directory = tempfile::tempdir().unwrap();
        let source_model = model(32);
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        let first =
            publish_qat_candidate(&source_model, directory.path(), "sealed", &recipe).unwrap();

        // The in-memory model is intentionally unrelated. The trusted input to
        // this retry is the digest from an immutable checkpoint generation.
        // A hook failure would prove that publication tried to serialize the
        // mutable model instead of taking the authenticated existing-key path.
        let unrelated_model = model(33);
        let reopened = publish_qat_candidate_with_hook(
            &unrelated_model,
            directory.path(),
            "sealed",
            &recipe,
            Some(&first.weights_sha256),
            |point, _| bail!("unexpected serialization boundary {point:?}"),
        )
        .unwrap();
        assert_eq!(reopened, first);

        let wrong_source = format!("sha256:{}", "0".repeat(64));
        let error = publish_qat_candidate_from_authenticated_source(
            &unrelated_model,
            directory.path(),
            "sealed",
            &recipe,
            &wrong_source,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("different model weights"), "{error}");
    }

    #[test]
    fn orphan_adoption_rejects_an_internally_rehashed_wrong_archive() {
        let directory = tempfile::tempdir().unwrap();
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        let source_model = model(32);
        let source =
            publish_qat_candidate(&source_model, directory.path(), "source", &recipe).unwrap();
        let wrong_model = model(33);
        let forged =
            publish_qat_candidate(&wrong_model, directory.path(), "forged", &recipe).unwrap();

        // Forge a candidate whose outer and inner hashes are all
        // self-consistent and claim the authenticated source identity, while
        // retaining quantized tensors derived from a different model.
        fs::copy(&source.weights_path, &forged.weights_path).unwrap();
        let mut archive_manifest: QuantizationManifest =
            read_json_regular(&forged.archive_manifest_path, "HQUANT manifest").unwrap();
        archive_manifest.base_checkpoint_hash = source.weights_sha256.clone();
        let archive_manifest_bytes = serde_json::to_vec_pretty(&archive_manifest).unwrap();
        fs::write(&forged.archive_manifest_path, &archive_manifest_bytes).unwrap();
        let forged_archive = QuantizedArchive::open(&forged.archive_path).unwrap();

        let mut candidate_manifest: QatCandidateManifest =
            read_json_regular(&forged.candidate_manifest_path, "QAT candidate manifest").unwrap();
        candidate_manifest.weights_bytes = fs::metadata(&source.weights_path).unwrap().len();
        candidate_manifest.weights_sha256 = source.weights_sha256.clone();
        candidate_manifest.archive_manifest_sha256 = sha256_label(&archive_manifest_bytes);
        candidate_manifest.archive_content_sha256 = forged_archive.content_hash().unwrap();
        fs::write(
            &forged.candidate_manifest_path,
            serde_json::to_vec_pretty(&candidate_manifest).unwrap(),
        )
        .unwrap();

        let error = open_qat_candidate(&forged.candidate_root).unwrap_err();
        assert!(
            format!("{error:#}").contains("does not reproduce its canonical weights"),
            "{error:#}"
        );
        let error = publish_qat_candidate_from_authenticated_source(
            &wrong_model,
            directory.path(),
            "forged",
            &recipe,
            &source.weights_sha256,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("does not reproduce its canonical weights"),
            "{error:#}"
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
                None,
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
    fn concurrent_adoption_requires_the_exact_locally_derived_receipt() {
        let directory = tempfile::tempdir().unwrap();
        let model = model(32);
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        let key = "concurrent-receipt";
        let competing = directory.path().join(key);
        let error = publish_qat_candidate_with_hook(
            &model,
            directory.path(),
            key,
            &recipe,
            None,
            |point, staging| {
                if point == PublicationPoint::ManifestSynced {
                    copy_tree(staging.path(), &competing);
                    let manifest: QatCandidateManifest = read_json_regular(
                        &competing.join(CANDIDATE_MANIFEST),
                        "competing candidate manifest",
                    )?;
                    // The candidate is semantically equivalent and internally
                    // valid, but it is not the exact receipt derived by this
                    // publication attempt.
                    fs::write(
                        competing.join(CANDIDATE_MANIFEST),
                        serde_json::to_vec(&manifest)?,
                    )?;
                }
                Ok(())
            },
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("different candidate bytes"),
            "{error:#}"
        );
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
        assert!(
            format!("{error:#}").contains("invalid HQUANT archive"),
            "{error:#}"
        );
        let fast_error = publish_qat_candidate_from_authenticated_source(
            &model,
            directory.path(),
            "tamper",
            &recipe,
            &publication.weights_sha256,
        )
        .unwrap_err();
        assert!(
            format!("{fast_error:#}").contains("invalid HQUANT archive"),
            "{fast_error:#}"
        );
        assert!(publication.candidate_root.exists());
    }

    #[test]
    fn candidate_root_swap_after_manifest_capture_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let first_store = directory.path().join("first-store");
        let second_store = directory.path().join("second-store");
        let recipe = recipe(UltraQuantFormat::BinaryG128);
        let first = publish_qat_candidate(&model(32), &first_store, "stable", &recipe).unwrap();
        let second = publish_qat_candidate(&model(33), &second_store, "stable", &recipe).unwrap();
        let published = first.candidate_root.clone();
        let replacement = second.candidate_root.clone();
        let parked = directory.path().join("parked-candidate");

        let error = validate_candidate_inner_with_hook(&published, "stable", false, None, || {
            fs::rename(&published, &parked)?;
            fs::rename(&replacement, &published)?;
            Ok(())
        })
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("weights do not match"),
            "{error:#}"
        );
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
        assert!(
            format!("{error:#}").contains("unverified top-level"),
            "{error:#}"
        );
    }
}
