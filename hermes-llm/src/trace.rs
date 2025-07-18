//! Bounded, portable model traces for the educational/debugging web lab.
//!
//! Tracing is deliberately opt-in. The normal inference and training paths do
//! not construct any of these values.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Int, Tensor, TensorData};
use serde::Serialize;
use serde_json::Value;

use crate::mal::{Activation, ModelDef, NormPosition, NormType, PositionEncoding};
use crate::{Tokenizer, Transformer};

pub const TRACE_KIND: &str = "hermes_model_trace";
pub const TRACE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct TraceOptions {
    pub token_limit: usize,
    pub channel_limit: usize,
    pub attention_head_limit: usize,
    pub metrics_row_limit: usize,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self {
            token_limit: 128,
            channel_limit: 64,
            attention_head_limit: 4,
            metrics_row_limit: 2_000,
        }
    }
}

impl TraceOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.token_limit > 0, "trace token limit must be positive");
        ensure!(
            self.channel_limit > 0,
            "trace channel limit must be positive"
        );
        ensure!(
            self.attention_head_limit > 0,
            "trace attention-head limit must be positive"
        );
        ensure!(
            self.metrics_row_limit >= 2,
            "trace metrics-row limit must be at least 2"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct VisualizationBundle {
    pub kind: &'static str,
    pub version: u32,
    pub model: ModelTrace,
    pub inference: InferenceTrace,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingTrace>,
    pub capture: CaptureMetadata,
}

impl VisualizationBundle {
    pub fn write_pretty(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create trace output directory {}",
                    parent.display()
                )
            })?;
        }
        let file = File::create(path)
            .with_context(|| format!("failed to create trace bundle {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)
            .with_context(|| format!("failed to write trace bundle {}", path.display()))
    }
}

#[derive(Debug, Serialize)]
pub struct ModelTrace {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub estimated_parameters: usize,
    pub tied_embeddings: bool,
    pub layers: Vec<ArchitectureLayer>,
}

#[derive(Debug, Serialize)]
pub struct ArchitectureLayer {
    /// One-based index, matching the layer labels shown in the lab.
    pub index: usize,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern_index: Option<usize>,
    pub mixer: MixerTrace,
    pub ffn_hidden_size: usize,
    pub ffn_activation: String,
    pub ffn_gated: bool,
    pub norm: String,
    pub norm_position: String,
    pub residual: bool,
    pub dropout: f64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MixerTrace {
    Attention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        window_size: Option<usize>,
        qk_norm: bool,
        position_encoding: String,
    },
    Mamba {
        state_dim: usize,
        conv_kernel: usize,
        expand: usize,
        dt_rank: usize,
        inner_size: usize,
    },
}

#[derive(Debug, Serialize)]
pub struct InferenceTrace {
    pub prompt: String,
    pub generated_text: String,
    pub full_text: String,
    pub original_token_count: usize,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub token_offset: usize,
    pub tokens: Vec<TokenTrace>,
    pub sampling: SamplingTrace,
    pub stages: Vec<InferenceStageTrace>,
}

#[derive(Debug, Serialize)]
pub struct SamplingTrace {
    pub max_new_tokens: usize,
    pub temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    pub repetition_penalty: f64,
    pub seed: u64,
    pub stop_at_eos: bool,
}

#[derive(Debug, Serialize)]
pub struct TokenTrace {
    pub original_index: usize,
    pub id: u32,
    pub piece: String,
    pub display: String,
    pub source: &'static str,
}

#[derive(Debug, Serialize)]
pub struct InferenceStageTrace {
    pub stage: &'static str,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixer: Option<&'static str>,
    pub activation: Heatmap,
    pub stats: TensorStats,
    pub token_rms: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_delta_rms: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mamba_state: Option<Heatmap>,
}

#[derive(Debug, Serialize)]
pub struct AttentionTrace {
    pub total_heads: usize,
    pub captured_heads: usize,
    pub heads: Vec<Heatmap>,
}

#[derive(Debug, Serialize)]
pub struct Heatmap {
    pub rows: usize,
    pub cols: usize,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub values: Vec<f32>,
    pub min: f32,
    pub max: f32,
    pub value_kind: &'static str,
    pub original_rows: usize,
    pub original_cols: usize,
}

#[derive(Debug, Serialize)]
pub struct TensorStats {
    pub mean: f32,
    pub rms: f32,
    pub min: f32,
    pub max: f32,
    pub abs_max: f32,
}

#[derive(Debug, Serialize)]
pub struct TrainingTrace {
    pub source: String,
    pub total_rows: usize,
    pub captured_rows: usize,
    pub dropped_rows: usize,
    pub sampling_stride: f64,
    pub rows: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct CaptureMetadata {
    pub requested_token_limit: usize,
    pub original_tokens: usize,
    pub captured_tokens: usize,
    pub dropped_leading_tokens: usize,
    pub tokens_truncated: bool,
    pub original_hidden_channels: usize,
    pub captured_hidden_channels: usize,
    pub channels_reduced: bool,
    pub requested_attention_head_limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_total_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_captured_rows: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TraceGeneration {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub repetition_penalty: f64,
    pub seed: u64,
    pub stop_at_eos: bool,
}

pub struct TraceRequest<'a> {
    pub prompt: &'a str,
    pub prompt_token_count: usize,
    pub output_tokens: &'a [u32],
    pub generation: TraceGeneration,
    pub metrics_path: Option<&'a Path>,
}

pub fn capture_bundle(
    model: &Transformer,
    tokenizer: &Tokenizer,
    request: TraceRequest<'_>,
    options: &TraceOptions,
) -> Result<VisualizationBundle> {
    let TraceRequest {
        prompt,
        prompt_token_count,
        output_tokens,
        generation,
        metrics_path,
    } = request;
    options.validate()?;
    ensure!(
        !output_tokens.is_empty(),
        "cannot trace an empty token sequence"
    );
    ensure!(
        prompt_token_count <= output_tokens.len(),
        "prompt token count {prompt_token_count} exceeds output token count {}",
        output_tokens.len()
    );

    let effective_limit = options.token_limit.min(model.config().max_seq_len);
    let captured_token_count = output_tokens.len().min(effective_limit);
    let token_offset = output_tokens.len() - captured_token_count;
    let captured_ids = &output_tokens[token_offset..];
    let tokens = captured_ids
        .iter()
        .enumerate()
        .map(|(local_index, &id)| {
            let original_index = token_offset + local_index;
            let piece = tokenizer.token_piece(id)?;
            let decoded_piece = tokenizer.display_piece(id)?;
            Ok(TokenTrace {
                original_index,
                id,
                display: visible_token(&decoded_piece),
                piece,
                source: if original_index < prompt_token_count {
                    "prompt"
                } else {
                    "generated"
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let token_labels = tokens
        .iter()
        .map(|token| format!("{}:{}", token.original_index, token.display))
        .collect::<Vec<_>>();

    let values = captured_ids.iter().map(|&id| i64::from(id)).collect();
    let input = Tensor::<2, Int>::from_data(
        TensorData::new(values, [1, captured_token_count]),
        &model.device(),
    );
    let raw = model.forward_diagnostic(input, options.attention_head_limit);

    let mut stages = Vec::with_capacity(model.config().num_layers + 2);
    let (embedding, mut previous) = activation_stage(
        ActivationStageInfo {
            stage: "embedding",
            label: "Token embedding".to_owned(),
            layer_index: None,
            mixer: None,
        },
        raw.embedding,
        None,
        &token_labels,
        options.channel_limit,
    )?;
    stages.push(embedding);

    for (index, layer) in raw.layers.into_iter().enumerate() {
        let block = model.config().block_for_layer(index);
        let mixer = if block.is_ssm() { "mamba" } else { "attention" };
        let (mut stage, current) = activation_stage(
            ActivationStageInfo {
                stage: "block",
                label: format!("Layer {} · {}", index + 1, mixer),
                layer_index: Some(index + 1),
                mixer: Some(mixer),
            },
            layer.activation,
            Some(&previous),
            &token_labels,
            options.channel_limit,
        )?;
        stage.attention = layer
            .attention_weights
            .map(|weights| {
                attention_trace(
                    weights,
                    layer
                        .total_attention_heads
                        .expect("attention diagnostics always record the original head count"),
                    &token_labels,
                )
            })
            .transpose()?;
        stage.mamba_state = layer
            .mamba_state
            .map(|state| mamba_heatmap(state, options.channel_limit))
            .transpose()?;
        stages.push(stage);
        previous = current;
    }

    let (final_norm, _) = activation_stage(
        ActivationStageInfo {
            stage: "final_norm",
            label: "Final norm".to_owned(),
            layer_index: None,
            mixer: None,
        },
        raw.final_norm,
        Some(&previous),
        &token_labels,
        options.channel_limit,
    )?;
    stages.push(final_norm);

    let config = model.config();
    let training = metrics_path
        .map(|path| load_training_trace(path, options.metrics_row_limit))
        .transpose()?;
    let metrics_total_rows = training.as_ref().map(|trace| trace.total_rows);
    let metrics_captured_rows = training.as_ref().map(|trace| trace.captured_rows);
    let generated_tokens = &output_tokens[prompt_token_count..];

    Ok(VisualizationBundle {
        kind: TRACE_KIND,
        version: TRACE_VERSION,
        model: model_trace(config),
        inference: InferenceTrace {
            prompt: prompt.to_owned(),
            generated_text: tokenizer.decode(generated_tokens, true)?,
            full_text: tokenizer.decode(output_tokens, true)?,
            original_token_count: output_tokens.len(),
            prompt_token_count,
            generated_token_count: generated_tokens.len(),
            token_offset,
            tokens,
            sampling: SamplingTrace {
                max_new_tokens: generation.max_new_tokens,
                temperature: generation.temperature,
                top_k: generation.top_k,
                repetition_penalty: generation.repetition_penalty,
                seed: generation.seed,
                stop_at_eos: generation.stop_at_eos,
            },
            stages,
        },
        training,
        capture: CaptureMetadata {
            requested_token_limit: options.token_limit,
            original_tokens: output_tokens.len(),
            captured_tokens: captured_token_count,
            dropped_leading_tokens: token_offset,
            tokens_truncated: token_offset > 0,
            original_hidden_channels: config.hidden_size,
            captured_hidden_channels: config.hidden_size.min(options.channel_limit),
            channels_reduced: config.hidden_size > options.channel_limit,
            requested_attention_head_limit: options.attention_head_limit,
            metrics_total_rows,
            metrics_captured_rows,
        },
    })
}

fn model_trace(config: &ModelDef) -> ModelTrace {
    let pattern_len = config
        .pattern
        .as_ref()
        .filter(|p| !p.is_empty())
        .map(Vec::len);
    let layers = (0..config.num_layers)
        .map(|index| {
            let block = config.block_for_layer(index);
            let mixer = match &block.ssm {
                Some(ssm) => MixerTrace::Mamba {
                    state_dim: ssm.state_dim,
                    conv_kernel: ssm.conv_kernel,
                    expand: ssm.expand,
                    dt_rank: config.dt_rank(ssm),
                    inner_size: ssm.expand * config.hidden_size,
                },
                None => MixerTrace::Attention {
                    num_heads: block.num_heads(),
                    num_kv_heads: block.num_kv_heads(),
                    head_dim: block.head_dim(config.hidden_size),
                    causal: block.attention.causal,
                    window_size: block.attention.window_size,
                    qk_norm: block.attention.qk_norm,
                    position_encoding: position_name(&block.attention.position_encoding),
                },
            };
            ArchitectureLayer {
                index: index + 1,
                name: block.name.clone(),
                pattern_index: pattern_len.map(|len| index % len),
                mixer,
                ffn_hidden_size: block.intermediate_size(config.hidden_size),
                ffn_activation: activation_name(block.ffn.activation).to_owned(),
                ffn_gated: block.ffn.gate,
                norm: norm_name(block.norm.norm_type).to_owned(),
                norm_position: norm_position_name(block.norm_position).to_owned(),
                residual: block.residual,
                dropout: block.dropout,
            }
        })
        .collect();
    ModelTrace {
        name: config.name.clone(),
        description: config.description.clone(),
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        hidden_size: config.hidden_size,
        num_layers: config.num_layers,
        estimated_parameters: config.estimated_params(),
        tied_embeddings: config.embeddings.tie_weights,
        layers,
    }
}

struct ActivationStageInfo {
    stage: &'static str,
    label: String,
    layer_index: Option<usize>,
    mixer: Option<&'static str>,
}

fn activation_stage(
    info: ActivationStageInfo,
    tensor: Tensor<3>,
    previous: Option<&[f32]>,
    token_labels: &[String],
    channel_limit: usize,
) -> Result<(InferenceStageTrace, Vec<f32>)> {
    let ActivationStageInfo {
        stage,
        label,
        layer_index,
        mixer,
    } = info;
    let [batch, rows, cols] = tensor.dims();
    ensure!(batch == 1, "activation trace expected batch 1, got {batch}");
    ensure!(
        rows == token_labels.len(),
        "activation row count {rows} does not match {} token labels",
        token_labels.len()
    );
    let values = tensor_values(tensor, &label)?;
    let stats = tensor_stats(&values)?;
    let token_rms = values.chunks_exact(cols).map(row_rms).collect::<Vec<_>>();
    let token_delta_rms = previous.map(|prior| {
        prior
            .chunks_exact(cols)
            .zip(values.chunks_exact(cols))
            .map(|(left, right)| {
                let mean_square = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| (b - a).powi(2))
                    .sum::<f32>()
                    / cols as f32;
                mean_square.sqrt()
            })
            .collect()
    });
    let activation = row_binned_heatmap(
        &values,
        rows,
        cols,
        channel_limit,
        token_labels.to_vec(),
        "signed_mean",
    )?;
    Ok((
        InferenceStageTrace {
            stage,
            label,
            layer_index,
            mixer,
            activation,
            stats,
            token_rms,
            token_delta_rms,
            attention: None,
            mamba_state: None,
        },
        values,
    ))
}

fn attention_trace(
    tensor: Tensor<4>,
    total_heads: usize,
    token_labels: &[String],
) -> Result<AttentionTrace> {
    let [batch, heads, rows, cols] = tensor.dims();
    ensure!(batch == 1, "attention trace expected batch 1, got {batch}");
    ensure!(
        rows == token_labels.len() && cols == token_labels.len(),
        "attention trace shape {rows}x{cols} does not match {} tokens",
        token_labels.len()
    );
    let values = tensor_values(tensor, "attention weights")?;
    let head_size = rows * cols;
    let mut heatmaps = Vec::with_capacity(heads);
    for head in 0..heads {
        let slice = &values[head * head_size..(head + 1) * head_size];
        let (min, max) = min_max(slice)?;
        heatmaps.push(Heatmap {
            rows,
            cols,
            row_labels: token_labels.to_vec(),
            col_labels: token_labels.to_vec(),
            values: slice.to_vec(),
            min,
            max,
            value_kind: "probability",
            original_rows: rows,
            original_cols: cols,
        });
    }
    Ok(AttentionTrace {
        total_heads,
        captured_heads: heads,
        heads: heatmaps,
    })
}

fn mamba_heatmap(tensor: Tensor<3>, channel_limit: usize) -> Result<Heatmap> {
    let [batch, channels, state_dim] = tensor.dims();
    ensure!(
        batch == 1,
        "Mamba state trace expected batch 1, got {batch}"
    );
    let values = tensor_values(tensor, "Mamba recurrent state")?;
    let bins = channels.min(channel_limit);
    let mut reduced = Vec::with_capacity(state_dim * bins);
    for state in 0..state_dim {
        for bin in 0..bins {
            let (start, end) = bin_bounds(bin, bins, channels);
            let mean = (start..end)
                .map(|channel| values[channel * state_dim + state])
                .sum::<f32>()
                / (end - start) as f32;
            reduced.push(mean);
        }
    }
    let (min, max) = min_max(&reduced)?;
    Ok(Heatmap {
        rows: state_dim,
        cols: bins,
        row_labels: (0..state_dim)
            .map(|index| format!("state {index}"))
            .collect(),
        col_labels: channel_labels(channels, bins),
        values: reduced,
        min,
        max,
        value_kind: "signed_mean",
        original_rows: state_dim,
        original_cols: channels,
    })
}

fn row_binned_heatmap(
    values: &[f32],
    rows: usize,
    cols: usize,
    channel_limit: usize,
    row_labels: Vec<String>,
    value_kind: &'static str,
) -> Result<Heatmap> {
    ensure!(
        values.len() == rows * cols,
        "heatmap received {} values for {rows}x{cols}",
        values.len()
    );
    let bins = cols.min(channel_limit);
    let mut reduced = Vec::with_capacity(rows * bins);
    for row in values.chunks_exact(cols) {
        for bin in 0..bins {
            let (start, end) = bin_bounds(bin, bins, cols);
            reduced.push(row[start..end].iter().sum::<f32>() / (end - start) as f32);
        }
    }
    let (min, max) = min_max(&reduced)?;
    Ok(Heatmap {
        rows,
        cols: bins,
        row_labels,
        col_labels: channel_labels(cols, bins),
        values: reduced,
        min,
        max,
        value_kind,
        original_rows: rows,
        original_cols: cols,
    })
}

fn tensor_values<const D: usize>(tensor: Tensor<D>, name: &str) -> Result<Vec<f32>> {
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .with_context(|| format!("failed to copy {name} to the CPU"))?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{name} contains non-finite values"
    );
    Ok(values)
}

fn tensor_stats(values: &[f32]) -> Result<TensorStats> {
    let (min, max) = min_max(values)?;
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let rms = (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32).sqrt();
    Ok(TensorStats {
        mean,
        rms,
        min,
        max,
        abs_max: min.abs().max(max.abs()),
    })
}

fn row_rms(values: &[f32]) -> f32 {
    (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32).sqrt()
}

fn min_max(values: &[f32]) -> Result<(f32, f32)> {
    let Some((&first, rest)) = values.split_first() else {
        bail!("cannot summarize an empty tensor");
    };
    ensure!(first.is_finite(), "tensor contains a non-finite value");
    let mut min = first;
    let mut max = first;
    for &value in rest {
        ensure!(value.is_finite(), "tensor contains a non-finite value");
        min = min.min(value);
        max = max.max(value);
    }
    Ok((min, max))
}

fn bin_bounds(bin: usize, bins: usize, columns: usize) -> (usize, usize) {
    let start = bin * columns / bins;
    let end = ((bin + 1) * columns / bins).max(start + 1);
    (start, end)
}

fn channel_labels(columns: usize, bins: usize) -> Vec<String> {
    (0..bins)
        .map(|bin| {
            let (start, end) = bin_bounds(bin, bins, columns);
            if end == start + 1 {
                start.to_string()
            } else {
                format!("{start}..{}", end - 1)
            }
        })
        .collect()
}

fn visible_token(piece: &str) -> String {
    let visible = piece
        .replace([' ', 'Ġ', '▁'], "·")
        .replace('\n', "↵")
        .replace('\r', "⏎")
        .replace('\t', "⇥");
    let mut chars = visible.chars();
    let short = chars.by_ref().take(18).collect::<String>();
    if chars.next().is_some() {
        format!("{short}…")
    } else if short.is_empty() {
        "∅".to_owned()
    } else {
        short
    }
}

fn load_training_trace(path: &Path, row_limit: usize) -> Result<TrainingTrace> {
    let total_rows = scan_metric_rows(path)?;
    ensure!(total_rows > 0, "metrics file {} is empty", path.display());
    let selected = selected_rows(total_rows, row_limit);
    let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
    let file = File::open(path)
        .with_context(|| format!("failed to reopen metrics file {}", path.display()))?;
    let mut rows = Vec::with_capacity(selected.len());
    let mut observed = 0usize;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read metrics line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        let value = parse_metric_line(path, line_index, &line)?;
        if selected_set.contains(&line_index) {
            rows.push(value);
        }
        observed += 1;
    }
    ensure!(
        observed == total_rows,
        "metrics file {} changed while tracing (first pass {total_rows} rows, second pass {observed}); retry",
        path.display()
    );
    let captured_rows = rows.len();
    Ok(TrainingTrace {
        source: path.display().to_string(),
        total_rows,
        captured_rows,
        dropped_rows: total_rows - captured_rows,
        sampling_stride: if captured_rows <= 1 {
            0.0
        } else {
            (total_rows - 1) as f64 / (captured_rows - 1) as f64
        },
        rows,
    })
}

fn scan_metric_rows(path: &Path) -> Result<usize> {
    let file = File::open(path)
        .with_context(|| format!("failed to open metrics file {}", path.display()))?;
    let mut count = 0;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read metrics line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        parse_metric_line(path, line_index, &line)?;
        count += 1;
    }
    Ok(count)
}

fn parse_metric_line(path: &Path, line_index: usize, line: &str) -> Result<Value> {
    ensure!(
        !line.trim().is_empty(),
        "blank metrics line {} in {}",
        line_index + 1,
        path.display()
    );
    let value: Value = serde_json::from_str(line).with_context(|| {
        format!(
            "invalid JSON on metrics line {} in {}",
            line_index + 1,
            path.display()
        )
    })?;
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!(
            "metrics line {} in {} is not a JSON object",
            line_index + 1,
            path.display()
        )
    })?;
    ensure!(
        object.get("step").and_then(Value::as_u64).is_some(),
        "metrics line {} in {} has no non-negative integer `step`",
        line_index + 1,
        path.display()
    );
    Ok(value)
}

fn selected_rows(total: usize, limit: usize) -> Vec<usize> {
    if total <= limit {
        return (0..total).collect();
    }
    (0..limit)
        .map(|index| index * (total - 1) / (limit - 1))
        .collect()
}

fn activation_name(value: Activation) -> &'static str {
    match value {
        Activation::SwiGLU => "swiglu",
        Activation::GELU => "gelu",
        Activation::SiLU => "silu",
        Activation::ReLU => "relu",
        Activation::GELUNew => "gelu_new",
        Activation::GELUTanh => "gelu_tanh",
    }
}

fn norm_name(value: NormType) -> &'static str {
    match value {
        NormType::RmsNorm => "rms_norm",
        NormType::LayerNorm => "layer_norm",
        NormType::None => "none",
    }
}

fn norm_position_name(value: NormPosition) -> &'static str {
    match value {
        NormPosition::Pre => "pre",
        NormPosition::Post => "post",
    }
}

fn position_name(value: &PositionEncoding) -> String {
    match value {
        PositionEncoding::Rope { theta, scaling } => scaling.map_or_else(
            || format!("rope(theta={theta})"),
            |scale| format!("rope(theta={theta}, scale={scale})"),
        ),
        PositionEncoding::Alibi { learned_slopes } => {
            format!("alibi(learned_slopes={learned_slopes})")
        }
        PositionEncoding::Learned { max_positions } => {
            format!("learned(max_positions={max_positions})")
        }
        PositionEncoding::None => "none".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_metric_rows_keep_both_ends() {
        assert_eq!(selected_rows(3, 5), vec![0, 1, 2]);
        let rows = selected_rows(11, 4);
        assert_eq!(rows, vec![0, 3, 6, 10]);
    }

    #[test]
    fn row_binning_is_bounded_and_signed() {
        let heatmap = row_binned_heatmap(
            &[1.0, 3.0, -2.0, -4.0, 2.0, 4.0, -1.0, -3.0],
            2,
            4,
            2,
            vec!["a".to_owned(), "b".to_owned()],
            "signed_mean",
        )
        .unwrap();
        assert_eq!(heatmap.values, vec![2.0, -3.0, 3.0, -2.0]);
        assert_eq!(heatmap.original_cols, 4);
        assert_eq!(heatmap.cols, 2);
    }

    #[test]
    fn visible_token_normalizes_tokenizer_whitespace_markers() {
        assert_eq!(visible_token("Ġretrieval"), "·retrieval");
        assert_eq!(visible_token("▁planning"), "·planning");
        assert_eq!(visible_token(" literal"), "·literal");
    }

    #[test]
    fn metric_capture_downsamples_observably_and_rejects_blank_rows() {
        let directory = tempfile::tempdir().unwrap();
        let metrics = directory.path().join("metrics.jsonl");
        std::fs::write(
            &metrics,
            (0..5)
                .map(|step| format!("{{\"step\":{step},\"loss\":{}}}\n", 5 - step))
                .collect::<String>(),
        )
        .unwrap();
        let captured = load_training_trace(&metrics, 3).unwrap();
        assert_eq!(captured.total_rows, 5);
        assert_eq!(captured.captured_rows, 3);
        assert_eq!(captured.dropped_rows, 2);
        assert_eq!(
            captured
                .rows
                .iter()
                .map(|row| row["step"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 2, 4]
        );

        std::fs::write(&metrics, "{\"step\":0}\n\n{\"step\":1}\n").unwrap();
        let error = load_training_trace(&metrics, 3).unwrap_err().to_string();
        assert!(error.contains("blank metrics line 2"), "{error}");
    }
}
