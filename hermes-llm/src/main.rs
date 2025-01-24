use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};
use tracing::{Level, info, warn};
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
    /// Train a new model from scratch or fine-tune from checkpoint
    Train {
        /// Path to training data file (required for multi-GPU, optional for single GPU)
        #[arg(short, long)]
        data: Option<String>,

        /// Path to tokenizer file
        #[arg(short, long)]
        tokenizer: String,

        /// Model configuration preset (nano, tiny, gpt2-small, gpt2-medium, llama-small)
        #[arg(short, long, default_value = "tiny")]
        model: String,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output: String,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Number of epochs
        #[arg(short, long, default_value = "1")]
        epochs: usize,

        /// Sequence length
        #[arg(long, default_value = "256")]
        seq_len: usize,

        /// Learning rate (use lower values like 1e-5 for fine-tuning)
        #[arg(long, default_value = "3e-4")]
        lr: f64,

        /// Number of GPUs (1 = single GPU, >1 = distributed with NCCL)
        #[arg(long, default_value = "1")]
        num_gpus: usize,

        /// Gradient accumulation steps
        #[arg(long, default_value = "1")]
        grad_accum: usize,

        /// Path to pre-trained checkpoint for fine-tuning (optional)
        #[arg(long)]
        checkpoint: Option<String>,

        /// Number of layers to freeze from the bottom (for fine-tuning)
        #[arg(long, default_value = "0")]
        freeze_layers: usize,

        /// Resume training from interrupted checkpoint
        #[arg(long)]
        resume: bool,

        // Internal flags for distributed training (set automatically)
        // usize::MAX means "launcher mode" - will spawn workers
        #[arg(long, hide = true, default_value_t = usize::MAX)]
        rank: usize,

        #[arg(long, hide = true, default_value = "nccl_id.txt")]
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

    /// Direct Preference Optimization (DPO) training
    Dpo {
        /// Path to preference pairs file (JSONL with chosen/rejected)
        #[arg(short, long)]
        data: String,

        /// Path to tokenizer file
        #[arg(short, long)]
        tokenizer: String,

        /// Path to SFT checkpoint to start from
        #[arg(short, long)]
        checkpoint: String,

        /// Path to model config
        #[arg(long)]
        config: String,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints-dpo")]
        output: String,

        /// Batch size
        #[arg(short, long, default_value = "4")]
        batch_size: usize,

        /// Number of epochs
        #[arg(short, long, default_value = "1")]
        epochs: usize,

        /// Learning rate
        #[arg(long, default_value = "5e-7")]
        lr: f64,

        /// DPO beta parameter (controls divergence from reference)
        #[arg(long, default_value = "0.1")]
        beta: f64,

        /// Maximum sequence length
        #[arg(long, default_value = "512")]
        max_len: usize,
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
            warn!("Unknown model config '{}', using tiny", name);
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
            batch_size,
            epochs,
            seq_len,
            lr,
            num_gpus,
            grad_accum,
            checkpoint,
            freeze_layers,
            resume,
            rank,
            comm_file,
        } => {
            // Launcher mode: spawn child processes for ranks 1..n
            // Launcher becomes rank 0 to keep TTY for progress bar
            let children_handle = if num_gpus > 1 && rank == usize::MAX {
                use std::process::{Command, Stdio};

                // CRITICAL: Set CUDA_VISIBLE_DEVICES=0 BEFORE any CUDA operations
                unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "0") };

                let data_path = data
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--data is required for multi-GPU training"))?;

                info!("=== Distributed Training ===");
                info!("GPUs: {}", num_gpus);
                info!("Model: {}", model);
                info!(
                    "Effective batch: {} ({} x {} x {})",
                    batch_size * grad_accum * num_gpus,
                    batch_size,
                    grad_accum,
                    num_gpus
                );

                let exe = std::env::current_exe()?;
                let _ = std::fs::remove_file(&comm_file);

                let mut children = Vec::new();
                for r in 1..num_gpus {
                    info!("Launching rank {} on GPU {}...", r, r);

                    let mut child_cmd = Command::new(&exe);
                    child_cmd
                        .env("CUDA_VISIBLE_DEVICES", r.to_string())
                        .arg("train")
                        .arg("--data")
                        .arg(data_path)
                        .arg("--tokenizer")
                        .arg(&tokenizer_path)
                        .arg("--model")
                        .arg(&model)
                        .arg("--output")
                        .arg(&output)
                        .arg("--batch-size")
                        .arg(batch_size.to_string())
                        .arg("--epochs")
                        .arg(epochs.to_string())
                        .arg("--seq-len")
                        .arg(seq_len.to_string())
                        .arg("--lr")
                        .arg(lr.to_string())
                        .arg("--num-gpus")
                        .arg(num_gpus.to_string())
                        .arg("--grad-accum")
                        .arg(grad_accum.to_string())
                        .arg("--freeze-layers")
                        .arg(freeze_layers.to_string())
                        .arg("--rank")
                        .arg(r.to_string())
                        .arg("--comm-file")
                        .arg(&comm_file);

                    if let Some(ref ckpt) = checkpoint {
                        child_cmd.arg("--checkpoint").arg(ckpt);
                    }

                    let child = child_cmd
                        .current_dir(std::env::current_dir()?)
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()?;

                    children.push((r, child));
                }

                std::thread::sleep(std::time::Duration::from_secs(2));

                // Spawn thread to wait for children
                Some(std::thread::spawn(move || {
                    let mut all_ok = true;
                    for (r, mut c) in children {
                        if !c.wait().map(|s| s.success()).unwrap_or(false) {
                            warn!("Rank {} failed", r);
                            all_ok = false;
                        }
                    }
                    all_ok
                }))
            } else {
                None
            };

            // Determine actual rank and device
            let actual_rank = if rank == usize::MAX { 0 } else { rank };
            let device = get_device(true, if num_gpus > 1 { 0 } else { actual_rank })?;

            let dist_config = hermes_llm::DistributedConfig {
                world_size: num_gpus,
                rank: actual_rank,
                comm_file,
            };

            if dist_config.is_distributed() {
                info!(
                    "Rank {}/{} on GPU {}",
                    actual_rank + 1,
                    num_gpus,
                    actual_rank
                );
            }
            info!("Using device: {:?}", device);

            // Load or train tokenizer
            let tokenizer = if std::path::Path::new(&tokenizer_path).exists() {
                info!("Loading tokenizer from {}", tokenizer_path);
                Tokenizer::from_file(&tokenizer_path)?
            } else {
                let data_path = data
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--data required to train tokenizer"))?;
                info!("Training new tokenizer...");
                BPETrainer::new(32000).train_from_files(&[data_path.as_str()], &tokenizer_path)?
            };
            info!("Tokenizer vocab size: {}", tokenizer.vocab_size());

            // Model config
            let mut config = get_config(&model);
            config.vocab_size = tokenizer.vocab_size();
            info!("Model config: {:?}", config);

            // Load dataset
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

            // Create data loader (distributed if multi-GPU)
            let mut train_loader = if num_gpus > 1 {
                DataLoader::new_distributed(dataset, batch_size, true, actual_rank, num_gpus)
            } else {
                DataLoader::new(dataset, batch_size, true)
            };
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

            // Barrier before training
            if let Some(ref c) = comm {
                info!("Waiting for all ranks to synchronize...");
                c.barrier()?;
                info!("All ranks synchronized");
            }

            // Initialize trainer
            let mut trainer = Trainer::new(config.clone(), training_config, device)?;

            // Load checkpoint or resume state
            let resume_state = if resume {
                // Try to load interrupted training state
                let state = trainer.load_training_state(&output)?;
                if state.global_step > 0 {
                    info!(
                        "Resuming from epoch {}, step {}, batch {}",
                        state.epoch + 1,
                        state.global_step,
                        state.batch_position
                    );
                    Some(state)
                } else {
                    None
                }
            } else if let Some(ref ckpt_path) = checkpoint {
                info!("Loading checkpoint: {}", ckpt_path);
                trainer.load_checkpoint(ckpt_path)?;
                None
            } else {
                None
            };

            if freeze_layers > 0 {
                info!("Freezing {} layers", freeze_layers);
                trainer.freeze_layers(freeze_layers)?;
            }

            // Sync model weights from rank 0 to all ranks
            if let Some(ref c) = comm {
                info!("Broadcasting model weights...");
                hermes_llm::distributed::sync_model(trainer.var_map(), c)?;
            }

            // Save config (rank 0 only)
            if dist_config.is_main_process() {
                config.save_json(&format!("{}/config.json", output))?;
                info!("Saved config to {}/config.json", output);
            }

            // Train (with resume support)
            let completed = trainer.train_resumable(
                &mut train_loader,
                None,
                Some(&output),
                comm.as_ref(),
                resume_state,
            )?;

            // Finalize NCCL communicator properly before exit
            let is_worker = children_handle.is_none() && dist_config.is_distributed();
            if let Some(c) = comm {
                c.barrier()?;
                c.finalize()?;
            }

            // Worker processes exit immediately to avoid ncclCommDestroy hang
            if is_worker {
                std::process::exit(0);
            }

            // Cleanup for launcher
            if let Some(handle) = children_handle {
                let all_ok = handle.join().unwrap_or(false);
                let _ = std::fs::remove_file(&dist_config.comm_file);
                if all_ok {
                    if completed {
                        println!("\n=== Training complete ===");
                    } else {
                        println!("\n=== Training interrupted, checkpoint saved ===");
                        println!("Resume with: hermes-llm train --resume --output {}", output);
                    }
                } else {
                    anyhow::bail!("Some worker processes failed");
                }
            } else if dist_config.is_main_process() {
                if completed {
                    info!("Training complete!");
                } else {
                    info!("Training interrupted, checkpoint saved to {}", output);
                    info!("Resume with: hermes-llm train --resume --output {}", output);
                }
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

        Commands::Dpo {
            data,
            tokenizer: tokenizer_path,
            checkpoint,
            config: config_path,
            output,
            batch_size,
            epochs,
            lr,
            beta,
            max_len,
        } => {
            let device = get_device(true, 0)?;
            info!("Using device: {:?}", device);

            let config = Config::from_json(&config_path)?;
            let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

            info!("Loading preference dataset from {}", data);
            let dataset = hermes_llm::dpo::PreferenceDataset::from_file(&data)?;

            info!("Initializing DPO trainer...");
            let mut trainer =
                hermes_llm::dpo::DpoTrainer::new(config, &checkpoint, device, lr, beta, max_len)?;

            std::fs::create_dir_all(&output)?;

            trainer.train(&dataset, &tokenizer, epochs, batch_size, Some(&output))?;

            info!("DPO training complete!");
        }
    }

    Ok(())
}
