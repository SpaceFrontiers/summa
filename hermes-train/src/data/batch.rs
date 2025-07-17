//! Fixed-shape objective samples and device tensor batches.

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub(crate) struct EncodedText {
    pub(crate) tokens: Vec<i64>,
    pub(crate) end_position: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum TrainingSample {
    Causal {
        tokens: Vec<i64>,
    },
    Supervised {
        tokens: Vec<i64>,
        loss_positions: Vec<usize>,
        truncated_tokens: usize,
    },
    Retrieval {
        query: EncodedText,
        /// Positive first, then explicit negatives.
        documents: Vec<EncodedText>,
        truncated_tokens: usize,
    },
}

impl TrainingSample {
    fn truncated_tokens(&self) -> usize {
        match self {
            Self::Causal { .. } => 0,
            Self::Supervised {
                truncated_tokens, ..
            }
            | Self::Retrieval {
                truncated_tokens, ..
            } => *truncated_tokens,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Causal { .. } => "causal_lm",
            Self::Supervised { .. } => "supervised_generation",
            Self::Retrieval { .. } => "contrastive_retrieval",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BatchStats {
    pub(crate) examples: usize,
    /// Tokens passed through the Transformer, including fixed-shape padding.
    pub(crate) compute_tokens: usize,
    /// Tokens that contribute to a language-model loss.
    pub(crate) supervised_tokens: usize,
    pub(crate) truncated_tokens: usize,
    pub(crate) retrieval_candidates: usize,
}

pub(crate) struct LanguageBatch {
    pub(crate) input_ids: Tensor<2, Int>,
    pub(crate) targets: Tensor<2, Int>,
    pub(crate) loss_positions: Option<Tensor<1, Int>>,
    pub(super) stats: BatchStats,
}

pub(crate) struct RetrievalBatch {
    pub(crate) query_ids: Tensor<2, Int>,
    pub(crate) query_end_positions: Tensor<1, Int>,
    pub(crate) document_ids: Tensor<2, Int>,
    pub(crate) document_end_positions: Tensor<1, Int>,
    pub(crate) labels: Tensor<1, Int>,
    pub(super) stats: BatchStats,
}

pub(crate) enum TrainingBatch {
    Language(Box<LanguageBatch>),
    Retrieval(Box<RetrievalBatch>),
}

impl TrainingBatch {
    pub(crate) fn stats(&self) -> BatchStats {
        match self {
            Self::Language(batch) => batch.stats,
            Self::Retrieval(batch) => batch.stats,
        }
    }
}

fn checked_token_count(rows: usize, seq_len: usize, label: &str) -> Result<usize> {
    rows.checked_mul(seq_len)
        .with_context(|| format!("{label} token count overflows usize"))
}

fn flat_position(row: usize, seq_len: usize, position: usize, label: &str) -> Result<i64> {
    let position = row
        .checked_mul(seq_len)
        .and_then(|offset| offset.checked_add(position))
        .with_context(|| format!("{label} position overflows usize"))?;
    i64::try_from(position).with_context(|| format!("{label} position exceeds i64"))
}

pub(crate) fn make_batch(
    samples: &[TrainingSample],
    seq_len: usize,
    device: &Device,
) -> Result<TrainingBatch> {
    let first = samples
        .first()
        .context("cannot make a training batch from zero samples")?;
    let truncated_tokens = samples.iter().try_fold(0usize, |total, sample| {
        total
            .checked_add(sample.truncated_tokens())
            .context("batch truncated-token count overflows usize")
    })?;
    match first {
        TrainingSample::Causal { .. } => {
            let batch_tokens = checked_token_count(samples.len(), seq_len, "causal batch")?;
            let sample_tokens = seq_len
                .checked_add(1)
                .context("causal sample length overflows usize")?;
            let mut inputs = Vec::with_capacity(batch_tokens);
            let mut targets = Vec::with_capacity(batch_tokens);
            for sample in samples {
                let TrainingSample::Causal { tokens } = sample else {
                    bail!(
                        "mixed `{}` and `{}` samples in one batch",
                        first.kind(),
                        sample.kind()
                    );
                };
                ensure!(
                    tokens.len() == sample_tokens,
                    "causal sample has {} tokens, expected {}",
                    tokens.len(),
                    sample_tokens
                );
                inputs.extend_from_slice(&tokens[..seq_len]);
                targets.extend_from_slice(&tokens[1..]);
            }
            let stats = BatchStats {
                examples: samples.len(),
                compute_tokens: batch_tokens,
                supervised_tokens: batch_tokens,
                truncated_tokens,
                retrieval_candidates: 0,
            };
            Ok(TrainingBatch::Language(Box::new(LanguageBatch {
                input_ids: Tensor::from_data(
                    TensorData::new(inputs, [samples.len(), seq_len]),
                    device,
                ),
                targets: Tensor::from_data(
                    TensorData::new(targets, [samples.len(), seq_len]),
                    device,
                ),
                loss_positions: None,
                stats,
            })))
        }
        TrainingSample::Supervised { .. } => {
            let batch_tokens = checked_token_count(samples.len(), seq_len, "supervised batch")?;
            let sample_tokens = seq_len
                .checked_add(1)
                .context("supervised sample length overflows usize")?;
            let mut inputs = Vec::with_capacity(batch_tokens);
            let mut targets = Vec::with_capacity(batch_tokens);
            let mut positions = Vec::new();
            for (row, sample) in samples.iter().enumerate() {
                let TrainingSample::Supervised {
                    tokens,
                    loss_positions,
                    ..
                } = sample
                else {
                    bail!(
                        "mixed `{}` and `{}` samples in one batch",
                        first.kind(),
                        sample.kind()
                    );
                };
                ensure!(
                    tokens.len() == sample_tokens,
                    "supervised sample has {} tokens, expected {}",
                    tokens.len(),
                    sample_tokens
                );
                ensure!(
                    loss_positions.iter().all(|position| *position < seq_len),
                    "supervised loss position exceeds sequence_length {seq_len}"
                );
                inputs.extend_from_slice(&tokens[..seq_len]);
                targets.extend_from_slice(&tokens[1..]);
                for position in loss_positions {
                    positions.push(flat_position(row, seq_len, *position, "supervised loss")?);
                }
            }
            ensure!(
                !positions.is_empty(),
                "supervised batch has no target tokens"
            );
            let supervised_tokens = positions.len();
            let stats = BatchStats {
                examples: samples.len(),
                compute_tokens: batch_tokens,
                supervised_tokens,
                truncated_tokens,
                retrieval_candidates: 0,
            };
            Ok(TrainingBatch::Language(Box::new(LanguageBatch {
                input_ids: Tensor::from_data(
                    TensorData::new(inputs, [samples.len(), seq_len]),
                    device,
                ),
                targets: Tensor::from_data(
                    TensorData::new(targets, [samples.len(), seq_len]),
                    device,
                ),
                loss_positions: Some(Tensor::from_data(
                    TensorData::new(positions, [supervised_tokens]),
                    device,
                )),
                stats,
            })))
        }
        TrainingSample::Retrieval { .. } => {
            let query_tokens = checked_token_count(samples.len(), seq_len, "retrieval query")?;
            let mut queries = Vec::with_capacity(query_tokens);
            let mut query_end_positions = Vec::with_capacity(samples.len());
            let mut documents = Vec::new();
            let mut document_end_positions = Vec::new();
            let mut labels = Vec::with_capacity(samples.len());
            for (query_row, sample) in samples.iter().enumerate() {
                let TrainingSample::Retrieval {
                    query,
                    documents: sample_documents,
                    ..
                } = sample
                else {
                    bail!(
                        "mixed `{}` and `{}` samples in one batch",
                        first.kind(),
                        sample.kind()
                    );
                };
                ensure!(
                    query.tokens.len() == seq_len && query.end_position < seq_len,
                    "retrieval query does not match sequence_length {seq_len}"
                );
                ensure!(
                    !sample_documents.is_empty(),
                    "retrieval sample has no positive document"
                );
                queries.extend_from_slice(&query.tokens);
                query_end_positions.push(flat_position(
                    query_row,
                    seq_len,
                    query.end_position,
                    "retrieval query endpoint",
                )?);
                labels.push(
                    i64::try_from(document_end_positions.len())
                        .context("retrieval label exceeds i64")?,
                );
                for document in sample_documents {
                    ensure!(
                        document.tokens.len() == seq_len && document.end_position < seq_len,
                        "retrieval document does not match sequence_length {seq_len}"
                    );
                    let document_row = document_end_positions.len();
                    documents.extend_from_slice(&document.tokens);
                    document_end_positions.push(flat_position(
                        document_row,
                        seq_len,
                        document.end_position,
                        "retrieval document endpoint",
                    )?);
                }
            }
            let candidates = document_end_positions.len();
            ensure!(
                candidates > 1,
                "contrastive retrieval batch has one document; increase batch_size or add explicit negatives"
            );
            let sequences = samples
                .len()
                .checked_add(candidates)
                .context("retrieval sequence count overflows usize")?;
            let stats = BatchStats {
                examples: samples.len(),
                compute_tokens: checked_token_count(sequences, seq_len, "retrieval batch")?,
                supervised_tokens: 0,
                truncated_tokens,
                retrieval_candidates: candidates,
            };
            Ok(TrainingBatch::Retrieval(Box::new(RetrievalBatch {
                query_ids: Tensor::from_data(
                    TensorData::new(queries, [samples.len(), seq_len]),
                    device,
                ),
                query_end_positions: Tensor::from_data(
                    TensorData::new(query_end_positions, [samples.len()]),
                    device,
                ),
                document_ids: Tensor::from_data(
                    TensorData::new(documents, [candidates, seq_len]),
                    device,
                ),
                document_end_positions: Tensor::from_data(
                    TensorData::new(document_end_positions, [candidates]),
                    device,
                ),
                labels: Tensor::from_data(TensorData::new(labels, [samples.len()]), device),
                stats,
            })))
        }
    }
}
