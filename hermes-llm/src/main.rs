use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

use hermes_llm::config::{Config, TrainingConfig};
use hermes_llm::data::{DataLoader, Dataset};
use hermes_llm::tokenizer::{BPETrainer, Tokenizer};
use hermes_llm::training::{TextGenerator, Trainer};

#[derive(Parser)]
#[command(name = "hermes-llm")]
#[command(about = "Train LLMs from scratch in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model from scratch
    Train {
        /// Path to training data file (reads from stdin if not provided)
        #[arg(short, long)]
        data: Option<String>,

        /// Path to tokenizer file (will train new one if not exists)
        #[arg(short, long)]
        tokenizer: String,

        /// Model configuration preset (nano, tiny, gpt2-small, gpt2-medium, llama-small)
        #[arg(short, long, default_value = "tiny")]
        model: String,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output: String,

        /// Learning rate
        #[arg(long, default_value = "3e-4")]
        lr: f64,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Number of epochs
        #[arg(short, long, default_value = "1")]
        epochs: usize,

        /// Sequence length
        #[arg(long, default_value = "256")]
        seq_len: usize,

        /// Use GPU (Metal on macOS, CUDA on Linux/Windows)
        #[arg(long, default_value = "true")]
        gpu: bool,

        /// GPU device index (for multi-GPU systems)
        #[arg(long, default_value = "0")]
        gpu_id: usize,

        /// Gradient accumulation steps
        #[arg(long, default_value = "1")]
        grad_accum: usize,

        /// World size for distributed training (number of GPUs)
        #[arg(long, default_value = "1")]
        world_size: usize,

        /// Rank for distributed training (0 to world_size-1)
        #[arg(long, default_value = "0")]
        rank: usize,

        /// Communication file for NCCL ID exchange
        #[arg(long, default_value = "nccl_id.txt")]
        comm_file: String,
    },

    /// Train a BPE tokenizer
    TrainTokenizer {
        /// Input text files for training
        #[arg(short, long, num_args = 1..)]
        input: Vec<String>,

        /// Output tokenizer path
        #[arg(short, long)]
        output: String,

        /// Vocabulary size
        #[arg(short, long, default_value = "32000")]
        vocab_size: usize,
    },

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

        /// Use GPU (Metal on macOS, CUDA on Linux/Windows)
        #[arg(long, default_value = "true")]
        gpu: bool,
    },

    /// Show model info
    Info {
        /// Model configuration preset
        #[arg(short, long, default_value = "gpt2-small")]
        model: String,
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

fn get_config(name: &str) -> Config {
    match name {
        "nano" => Config::nano(),
        "tiny" => Config::tiny(),
        "gpt2-small" => Config::gpt2_small(),
        "gpt2-medium" => Config::gpt2_medium(),
        "gpt2-large" => Config::gpt2_large(),
        "llama-small" => Config::llama_small(),
        _ => {
            eprintln!("Unknown model config '{}', using tiny", name);
            Config::tiny()
        }
    }
}

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            data,
            tokenizer: tokenizer_path,
            model,
            output,
            lr,
            batch_size,
            epochs,
            seq_len,
            gpu,
            gpu_id,
            grad_accum,
            world_size,
            rank,
            comm_file,
        } => {
            // For distributed training, use rank as GPU ID
            let actual_gpu_id = if world_size > 1 { rank } else { gpu_id };
            let device = get_device(gpu, actual_gpu_id)?;

            // Setup distributed config
            let dist_config = hermes_llm::DistributedConfig {
                world_size,
                rank,
                comm_file,
            };

            if dist_config.is_distributed() {
                info!("Distributed training: rank {}/{}", rank, world_size);
            }
            info!("Using GPU {}", actual_gpu_id);
            info!("Using device: {:?}", device);

            let tokenizer = if std::path::Path::new(&tokenizer_path).exists() {
                info!("Loading tokenizer from {}", tokenizer_path);
                Tokenizer::from_file(&tokenizer_path)?
            } else {
                if data.is_none() {
                    anyhow::bail!(
                        "Cannot train tokenizer from stdin. Provide --data or pre-trained --tokenizer"
                    );
                }
                info!("Training new tokenizer...");
                let trainer = BPETrainer::new(32000);
                trainer.train_from_files(&[data.as_ref().unwrap().as_str()], &tokenizer_path)?
            };

            info!("Tokenizer vocab size: {}", tokenizer.vocab_size());

            let mut config = get_config(&model);
            config.vocab_size = tokenizer.vocab_size();

            info!("Model config: {:?}", config);

            let dataset = match &data {
                Some(path) => {
                    info!("Loading dataset from {}", path);
                    Dataset::from_file(path, &tokenizer, seq_len)?
                }
                None => {
                    info!("Loading dataset from stdin...");
                    Dataset::from_stdin(&tokenizer, seq_len)?
                }
            };
            info!("Dataset size: {} tokens", dataset.tokens().len());

            let mut train_loader = DataLoader::new(dataset, batch_size, true);
            info!("Number of batches: {}", train_loader.num_batches());

            let training_config = TrainingConfig {
                learning_rate: lr,
                batch_size,
                epochs,
                seq_len,
                gradient_accumulation_steps: grad_accum,
                ..Default::default()
            };

            std::fs::create_dir_all(&output)?;

            // Initialize NCCL communicator for distributed training
            #[cfg(feature = "nccl")]
            let comm = if dist_config.is_distributed() {
                Some(hermes_llm::NcclCommunicator::new(&dist_config)?)
            } else {
                None
            };
            #[cfg(not(feature = "nccl"))]
            let comm: Option<hermes_llm::NcclCommunicator> = None;

            if dist_config.is_distributed() && comm.is_none() {
                anyhow::bail!("Distributed training requires --features nccl");
            }

            let mut trainer = Trainer::new(config.clone(), training_config, device)?;

            // Only rank 0 saves config
            if dist_config.is_main_process() {
                config.save_json(&format!("{}/config.json", output))?;
                info!("Saved config to {}/config.json", output);
            }

            trainer.train_distributed(&mut train_loader, None, Some(&output), comm.as_ref())?;

            if dist_config.is_main_process() {
                info!("Training complete!");
            }
        }

        Commands::TrainTokenizer {
            input,
            output,
            vocab_size,
        } => {
            info!("Training BPE tokenizer with vocab size {}", vocab_size);
            let trainer = BPETrainer::new(vocab_size);
            let files: Vec<&str> = input.iter().map(|s| s.as_str()).collect();
            let tokenizer = trainer.train_from_files(&files, &output)?;
            info!(
                "Tokenizer trained and saved to {} (vocab size: {})",
                output,
                tokenizer.vocab_size()
            );
        }

        Commands::Generate {
            checkpoint,
            config: config_path,
            tokenizer: tokenizer_path,
            prompt,
            max_tokens,
            temperature,
            top_k,
            gpu,
        } => {
            let device = get_device(gpu, 0)?;
            info!("Using device: {:?}", device);

            let config = Config::from_json(&config_path)?;
            let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

            let mut var_map = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
            let model = hermes_llm::GPT::new(&config, vb)?;
            var_map.load(&checkpoint)?;

            info!("Loaded model from {}", checkpoint);

            let prompt_tokens = tokenizer.encode(&prompt, false)?;
            info!("Prompt tokens: {:?}", prompt_tokens);

            let generator = TextGenerator::new(&model, &device);
            let output_tokens =
                generator.generate(&prompt_tokens, max_tokens, temperature, top_k)?;

            let output_text = tokenizer.decode(&output_tokens, true)?;
            println!("\n{}", output_text);
        }

        Commands::Info { model } => {
            let config = get_config(&model);
            println!("Model: {}", model);
            println!("  Vocab size: {}", config.vocab_size);
            println!("  Max sequence length: {}", config.max_seq_len);
            println!("  Hidden size: {}", config.hidden_size);
            println!("  Num layers: {}", config.num_layers);
            println!("  Num heads: {}", config.num_heads);
            println!("  Intermediate size: {}", config.intermediate_size);
            println!("  Head dimension: {}", config.head_dim());

            let dummy_config = config.clone();
            let embed_params = dummy_config.vocab_size * dummy_config.hidden_size;
            let attn_params = 4 * dummy_config.hidden_size * dummy_config.hidden_size;
            let ff_params = 3 * dummy_config.hidden_size * dummy_config.intermediate_size;
            let layer_params = attn_params + ff_params + 2 * dummy_config.hidden_size;
            let head_params = dummy_config.hidden_size * dummy_config.vocab_size;
            let total = embed_params + dummy_config.num_layers * layer_params + head_params;
            println!(
                "  Estimated parameters: {} ({:.2}M)",
                total,
                total as f64 / 1_000_000.0
            );
        }
    }

    Ok(())
}
