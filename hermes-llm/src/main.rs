#[cfg(feature = "lab")]
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{Level, info, warn};
use tracing_subscriber::FmtSubscriber;

use hermes_llm::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "hermes-llm")]
#[command(about = "Hermes LLM inference and model definition tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a trained model
    Generate {
        /// Model weights path or remote URI
        #[arg(short, long)]
        checkpoint: String,

        /// Model config path or remote URI
        #[arg(long)]
        config: String,

        /// Tokenizer path or remote URI
        #[arg(short, long)]
        tokenizer: String,

        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,

        /// Sampling temperature
        #[arg(long, default_value = "0.8")]
        temperature: f64,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<usize>,

        /// RNG seed for reproducible sampling (random if unset)
        #[arg(long)]
        seed: Option<u64>,

        /// Keep generating for the full --max-tokens instead of stopping at EOS
        #[arg(long, default_value_t = false)]
        no_eos: bool,
    },

    /// Generate text and export a bounded model-visualization trace
    Trace {
        /// Model weights path or remote URI
        #[arg(short, long)]
        checkpoint: String,

        /// MAL source or exported JSON model config path/URI
        #[arg(long)]
        config: String,

        /// Tokenizer path or remote URI
        #[arg(short, long)]
        tokenizer: String,

        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// JSON trace bundle output
        #[arg(short, long)]
        output: PathBuf,

        /// Optional hermes-train metrics.jsonl to include
        #[arg(long)]
        metrics: Option<PathBuf>,

        /// Maximum number of tokens to generate
        #[arg(short, long, default_value = "32")]
        max_tokens: usize,

        /// Sampling temperature; <= 0 uses greedy decoding
        #[arg(long, default_value = "0.8")]
        temperature: f64,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<usize>,

        /// RNG seed for reproducible sampling (random if unset)
        #[arg(long)]
        seed: Option<u64>,

        /// Keep generating for the full --max-tokens instead of stopping at EOS
        #[arg(long, default_value_t = false)]
        no_eos: bool,

        /// Maximum trailing tokens captured by the diagnostic pass
        #[arg(long, default_value_t = 128)]
        trace_tokens: usize,

        /// Maximum residual/Mamba channel bins per heatmap
        #[arg(long, default_value_t = 64)]
        channel_bins: usize,

        /// Maximum attention heads captured per attention layer
        #[arg(long, default_value_t = 4)]
        attention_heads: usize,

        /// Maximum training metric rows retained in the bundle
        #[arg(long, default_value_t = 2_000)]
        metrics_points: usize,
    },

    /// Keep a checkpoint loaded and serve the interactive Model Lab locally
    #[cfg(feature = "lab")]
    Lab {
        /// Model weights path or remote URI
        #[arg(short, long)]
        checkpoint: String,

        /// MAL source or exported JSON model config path/URI
        #[arg(long)]
        config: String,

        /// Tokenizer path or remote URI
        #[arg(short, long)]
        tokenizer: String,

        /// Optional hermes-train metrics.jsonl included with every trace
        #[arg(long)]
        metrics: Option<PathBuf>,

        /// Directory containing model-lab.html and src/model-lab
        #[arg(long, default_value = "hermes-web")]
        web_root: PathBuf,

        /// HTTP address; loopback is required unless --allow-remote is set
        #[arg(long, default_value = "127.0.0.1:4173")]
        bind: SocketAddr,

        /// Explicitly allow exposing prompts and traces on a non-loopback address
        #[arg(long, default_value_t = false)]
        allow_remote: bool,

        /// Maximum generated tokens accepted from one browser request
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,

        /// Maximum UTF-8 prompt bytes accepted from one browser request
        #[arg(long, default_value_t = 16 * 1024)]
        max_prompt_bytes: usize,

        /// Maximum trailing tokens captured by the diagnostic pass
        #[arg(long, default_value_t = 128)]
        trace_tokens: usize,

        /// Maximum residual/Mamba channel bins per heatmap
        #[arg(long, default_value_t = 64)]
        channel_bins: usize,

        /// Maximum attention heads captured per attention layer
        #[arg(long, default_value_t = 4)]
        attention_heads: usize,

        /// Maximum training metric rows retained in each response
        #[arg(long, default_value_t = 2_000)]
        metrics_points: usize,
    },

    /// Show model info
    Info {
        /// Model configuration preset
        #[arg(short, long, default_value = "gpt2-small")]
        model: String,
    },

    /// Export a MAL model definition as JSON
    Export {
        /// Model configuration: preset name (nano, tiny, gpt2-small, llama-7b) or path to .mal file
        #[arg(short, long)]
        model: String,

        /// Output JSON path (prints to stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn get_model_def(name: &str) -> Result<hermes_llm::ModelDef> {
    // First try builtin models
    if let Some(model_def) = hermes_llm::get_builtin_model(name) {
        return Ok(model_def);
    }

    // Try to load from .mal file if it exists — surface the real parse error
    // rather than a misleading "unknown model".
    if std::path::Path::new(name).exists() {
        return hermes_llm::parse_mal_file(name)
            .with_context(|| format!("failed to parse MAL file '{name}'"));
    }

    anyhow::bail!(
        "Unknown model '{}'. Available: {:?}",
        name,
        hermes_llm::list_wellknown_models()
    );
}

fn load_model_def(path: &Path) -> Result<hermes_llm::ModelDef> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mal"))
    {
        return hermes_llm::parse_mal_file(path)
            .with_context(|| format!("failed to parse MAL config {}", path.display()));
    }
    hermes_llm::ModelDef::from_json(path)
        .with_context(|| format!("failed to parse JSON model config {}", path.display()))
}

fn load_inference_artifacts(
    checkpoint: &str,
    config: &str,
    tokenizer: &str,
) -> Result<(
    hermes_llm::Transformer,
    hermes_llm::Device,
    Tokenizer,
    PathBuf,
)> {
    let device = hermes_llm::default_device();
    info!("Using device: {:?}", device);

    let checkpoint_path = hermes_llm::remote::resolve(checkpoint)?;
    let config_path = hermes_llm::remote::resolve(config)?;
    let tokenizer_path = hermes_llm::remote::resolve(tokenizer)?;
    let config = load_model_def(&config_path)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    anyhow::ensure!(
        config.vocab_size == tokenizer.vocab_size(),
        "model vocab_size {} does not match tokenizer vocab_size {}",
        config.vocab_size,
        tokenizer.vocab_size()
    );

    let mut model = hermes_llm::Transformer::new(&config, &device)?;
    hermes_llm::load_safetensors(&mut model, &checkpoint_path)?;
    model.prepare_inference();
    Ok((model, device, tokenizer, checkpoint_path))
}

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            checkpoint,
            config: config_path,
            tokenizer: tokenizer_path,
            prompt,
            max_tokens,
            temperature,
            top_k,
            seed,
            no_eos,
        } => {
            let (model, device, tokenizer, checkpoint_path) =
                load_inference_artifacts(&checkpoint, &config_path, &tokenizer_path)?;

            info!("Loaded model from {}", checkpoint_path.display());

            let prompt_tokens = tokenizer.encode(&prompt, false)?;
            info!("Prompt tokens: {:?}", prompt_tokens);

            let generator = hermes_llm::TextGenerator::new(&model, &device);
            let sampling = hermes_llm::generate::SamplingConfig {
                max_new_tokens: max_tokens,
                temperature,
                top_k,
                eos_token: (!no_eos).then(|| tokenizer.eos_token_id()),
                seed,
            };
            let output_tokens = generator.generate(&prompt_tokens, &sampling)?;

            let output_text = tokenizer.decode(&output_tokens, true)?;
            println!("\n{}", output_text);
        }

        Commands::Trace {
            checkpoint,
            config,
            tokenizer: tokenizer_path,
            prompt,
            output,
            metrics,
            max_tokens,
            temperature,
            top_k,
            seed,
            no_eos,
            trace_tokens,
            channel_bins,
            attention_heads,
            metrics_points,
        } => {
            let (model, device, tokenizer, checkpoint_path) =
                load_inference_artifacts(&checkpoint, &config, &tokenizer_path)?;
            info!("Loaded model from {}", checkpoint_path.display());
            let prompt_tokens = tokenizer.encode(&prompt, false)?;
            anyhow::ensure!(!prompt_tokens.is_empty(), "prompt encodes to zero tokens");
            let actual_seed = seed.unwrap_or_else(rand::random);
            let sampling = hermes_llm::generate::SamplingConfig {
                max_new_tokens: max_tokens,
                temperature,
                top_k,
                eos_token: (!no_eos).then(|| tokenizer.eos_token_id()),
                seed: Some(actual_seed),
            };
            let generator = hermes_llm::TextGenerator::new(&model, &device);
            let output_tokens = generator.generate(&prompt_tokens, &sampling)?;
            info!(
                "Generation complete; running opt-in full-sequence diagnostic pass over at most {trace_tokens} tokens"
            );
            let options = hermes_llm::TraceOptions {
                token_limit: trace_tokens,
                channel_limit: channel_bins,
                attention_head_limit: attention_heads,
                metrics_row_limit: metrics_points,
            };
            let bundle = hermes_llm::capture_bundle(
                &model,
                &tokenizer,
                hermes_llm::TraceRequest {
                    prompt: &prompt,
                    prompt_token_count: prompt_tokens.len(),
                    output_tokens: &output_tokens,
                    generation: hermes_llm::TraceGeneration {
                        max_new_tokens: max_tokens,
                        temperature,
                        top_k,
                        seed: actual_seed,
                        stop_at_eos: !no_eos,
                    },
                    metrics_path: metrics.as_deref(),
                },
                &options,
            )?;

            if bundle.capture.tokens_truncated {
                warn!(
                    "Trace retained {} of {} tokens and dropped {} leading tokens",
                    bundle.capture.captured_tokens,
                    bundle.capture.original_tokens,
                    bundle.capture.dropped_leading_tokens
                );
            }
            if bundle.capture.channels_reduced {
                info!(
                    "Trace reduced {} hidden channels into {} contiguous bins",
                    bundle.capture.original_hidden_channels,
                    bundle.capture.captured_hidden_channels
                );
            }
            for stage in &bundle.inference.stages {
                if let Some(attention) = &stage.attention
                    && attention.captured_heads < attention.total_heads
                {
                    info!(
                        "{} retained {} of {} attention heads",
                        stage.label, attention.captured_heads, attention.total_heads
                    );
                }
            }
            if let Some(training) = &bundle.training
                && training.dropped_rows > 0
            {
                info!(
                    "Trace retained {} of {} training metric rows (stride {:.2})",
                    training.captured_rows, training.total_rows, training.sampling_stride
                );
            }
            bundle.write_pretty(&output)?;
            info!("Wrote model trace to {}", output.display());
            println!("\n{}", bundle.inference.full_text);
        }

        #[cfg(feature = "lab")]
        Commands::Lab {
            checkpoint,
            config,
            tokenizer: tokenizer_path,
            metrics,
            web_root,
            bind,
            allow_remote,
            max_new_tokens,
            max_prompt_bytes,
            trace_tokens,
            channel_bins,
            attention_heads,
            metrics_points,
        } => {
            let (model, device, tokenizer, checkpoint_path) =
                load_inference_artifacts(&checkpoint, &config, &tokenizer_path)?;
            info!("Loaded model from {}", checkpoint_path.display());
            let server = hermes_llm::lab::LabServerConfig {
                bind,
                allow_remote,
                web_root,
                metrics_path: metrics,
                trace_options: hermes_llm::TraceOptions {
                    token_limit: trace_tokens,
                    channel_limit: channel_bins,
                    attention_head_limit: attention_heads,
                    metrics_row_limit: metrics_points,
                },
                max_new_tokens,
                max_prompt_bytes,
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("failed to create Model Lab runtime")?
                .block_on(hermes_llm::lab::serve_lab(model, device, tokenizer, server))?;
        }

        Commands::Info { model } => {
            let model_def = get_model_def(&model)?;
            print!("{}", model_def);
        }

        Commands::Export { model, output } => {
            let model_def = get_model_def(&model)?;
            match output {
                Some(path) => {
                    model_def.save_json(&path)?;
                    info!("Exported model config to {}", path);
                }
                None => {
                    println!("{}", serde_json::to_string_pretty(&model_def)?);
                }
            }
        }
    }

    Ok(())
}
