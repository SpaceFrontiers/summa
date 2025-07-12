use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand};
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

use hermes_llm::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "hermes-llm")]
#[command(
    about = "Hermes LLM inference and model definition tool (training lives in hermes-train)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a trained model
    Generate {
        /// Path to model checkpoint
        #[arg(short, long)]
        checkpoint: String,

        /// Path to model config
        #[arg(long)]
        config: String,

        /// Path to tokenizer
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

        /// Use GPU (Metal on macOS, CUDA on Linux/Windows); pass `--gpu false` for CPU
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        gpu: bool,
    },

    /// Show model info
    Info {
        /// Model configuration preset
        #[arg(short, long, default_value = "gpt2-small")]
        model: String,
    },

    /// Export a MAL model definition as JSON (consumed by hermes-train)
    Export {
        /// Model configuration: preset name (nano, tiny, gpt2-small, llama-7b) or path to .mal file
        #[arg(short, long)]
        model: String,

        /// Output JSON path (prints to stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[allow(unused_variables)]
fn get_device(use_gpu: bool, gpu_id: usize) -> Result<Device> {
    if use_gpu {
        #[cfg(feature = "metal")]
        {
            return Ok(Device::new_metal(gpu_id)?);
        }
        #[cfg(feature = "cuda")]
        {
            return Ok(Device::new_cuda(gpu_id)?);
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            tracing::warn!(
                "No GPU feature enabled, using CPU. Build with --features metal or --features cuda"
            );
            return Ok(Device::Cpu);
        }
    }
    Ok(Device::Cpu)
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
            gpu,
        } => {
            let device = get_device(gpu, 0)?;
            info!("Using device: {:?}", device);

            // Each artifact may be a local path or a remote URI (s3://, gs://,
            // http(s)://); resolve downloads + caches remote ones.
            let checkpoint_path = hermes_llm::remote::resolve(&checkpoint)?;
            let config_path = hermes_llm::remote::resolve(&config_path)?;
            let tokenizer_path = hermes_llm::remote::resolve(&tokenizer_path)?;

            let config = hermes_llm::ModelDef::from_json(&config_path)?;
            let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

            let mut var_map = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
            let model = hermes_llm::Transformer::new(&config, vb)?;
            var_map.load(&checkpoint_path)?;

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
