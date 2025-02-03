use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::info;

use crate::config::TrainingConfig;
use crate::data::DataLoader;
use crate::mal::ModelDef;
use crate::model::{Transformer, cross_entropy_loss};

/// Create a styled progress bar for training.
pub fn create_progress_bar(total: u64, show: bool) -> ProgressBar {
    if show {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} loss: {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    } else {
        ProgressBar::hidden()
    }
}

/// Training state for checkpointing and resume
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingState {
    pub epoch: usize,
    pub batch_position: usize,
    pub global_step: usize,
    pub shuffle_seed: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            batch_position: 0,
            global_step: 0,
            shuffle_seed: 42,
        }
    }
}

pub struct Trainer {
    model: Transformer,
    optimizer: AdamW,
    var_map: VarMap,
    #[allow(dead_code)]
    config: ModelDef,
    training_config: TrainingConfig,
    device: Device,
    global_step: usize,
    /// Signal for graceful shutdown on Ctrl+C
    interrupted: Arc<AtomicBool>,
}

impl Trainer {
    pub fn new(config: ModelDef, training_config: TrainingConfig, device: Device) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb)?;

        let params = ParamsAdamW {
            lr: training_config.learning_rate,
            beta1: training_config.beta1,
            beta2: training_config.beta2,
            weight_decay: training_config.weight_decay,
            eps: 1e-8,
        };
        let optimizer = AdamW::new(var_map.all_vars(), params)?;

        info!(
            "Initialized model with {} parameters",
            model.num_parameters()
        );

        // Set up Ctrl+C handler
        let interrupted = Arc::new(AtomicBool::new(false));
        let interrupted_clone = Arc::clone(&interrupted);
        let _ = ctrlc::set_handler(move || {
            // Use eprintln here since tracing may not be flushed before exit
            eprintln!("\nInterrupt received, saving checkpoint...");
            interrupted_clone.store(true, Ordering::SeqCst);
        });

        Ok(Self {
            model,
            optimizer,
            var_map,
            config,
            training_config,
            device,
            global_step: 0,
            interrupted,
        })
    }

    /// Check if training was interrupted
    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }

    /// Reset interrupt flag
    pub fn clear_interrupt(&self) {
        self.interrupted.store(false, Ordering::SeqCst);
    }

    /// Freeze the first N layers (embeddings count as layer 0)
    /// Frozen layers will not be updated during training
    pub fn freeze_layers(&mut self, num_layers: usize) -> Result<()> {
        let frozen_prefixes: Vec<String> =
            (0..num_layers).map(|i| format!("layers.{}", i)).collect();

        // Also freeze embeddings if num_layers > 0
        let mut prefixes = frozen_prefixes;
        if num_layers > 0 {
            prefixes.push("tok_emb".to_string());
        }

        let mut frozen_count = 0;
        for (name, var) in self.var_map.data().lock().unwrap().iter() {
            for prefix in &prefixes {
                if name.starts_with(prefix) {
                    // Detach tensor to prevent gradient computation
                    let tensor = var.as_tensor();
                    let _ = var.set(&tensor.detach());
                    frozen_count += 1;
                    break;
                }
            }
        }

        info!("Frozen {} parameter tensors", frozen_count);
        Ok(())
    }

    pub fn train_epoch(&mut self, train_loader: &mut DataLoader) -> Result<f64> {
        self.train_epoch_distributed(train_loader, None)
    }

    pub fn train_epoch_distributed(
        &mut self,
        train_loader: &mut DataLoader,
        comm: Option<&crate::distributed::NcclCommunicator>,
    ) -> Result<f64> {
        train_loader.reset();
        let (loss, _interrupted) = self.train_epoch_interruptible(train_loader, comm)?;
        Ok(loss)
    }

    pub fn evaluate(&self, eval_loader: &mut DataLoader) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        eval_loader.reset();

        while let Some(batch_result) = eval_loader.next_batch(&self.device)? {
            let (input, target) = (batch_result.0, batch_result.1);

            let logits = self.model.forward(&input, 0, false)?;
            let loss = cross_entropy_loss(&logits, &target)?;

            total_loss += loss.to_scalar::<f32>()? as f64;
            num_batches += 1;
        }

        if num_batches > 0 {
            Ok(total_loss / num_batches as f64)
        } else {
            Ok(0.0)
        }
    }

    pub fn train(
        &mut self,
        train_loader: &mut DataLoader,
        eval_loader: Option<&mut DataLoader>,
        checkpoint_dir: Option<&str>,
    ) -> Result<bool> {
        self.train_distributed(train_loader, eval_loader, checkpoint_dir, None)
    }

    pub fn train_distributed(
        &mut self,
        train_loader: &mut DataLoader,
        eval_loader: Option<&mut DataLoader>,
        checkpoint_dir: Option<&str>,
        comm: Option<&crate::distributed::NcclCommunicator>,
    ) -> Result<bool> {
        self.train_resumable(train_loader, eval_loader, checkpoint_dir, comm, None)
    }

    /// Train with support for interruption and resume
    /// Returns true if training completed, false if interrupted
    pub fn train_resumable(
        &mut self,
        train_loader: &mut DataLoader,
        mut eval_loader: Option<&mut DataLoader>,
        checkpoint_dir: Option<&str>,
        comm: Option<&crate::distributed::NcclCommunicator>,
        resume_state: Option<TrainingState>,
    ) -> Result<bool> {
        let is_main = comm.is_none_or(|c| c.rank() == 0);

        // Resume from saved state if provided
        let (start_epoch, start_position) = match resume_state {
            Some(ref state) => {
                self.global_step = state.global_step;
                if is_main {
                    info!(
                        "Resuming from epoch {}, batch position {}, global step {}",
                        state.epoch + 1,
                        state.batch_position,
                        state.global_step
                    );
                }
                (state.epoch, state.batch_position)
            }
            None => (0, 0),
        };

        if is_main {
            info!(
                "Starting training for {} epochs",
                self.training_config.epochs
            );
        }

        for epoch in start_epoch..self.training_config.epochs {
            if is_main {
                info!("Epoch {}/{}", epoch + 1, self.training_config.epochs);
            }

            // Use epoch as shuffle seed for reproducibility
            let shuffle_seed = epoch as u64;
            train_loader.reset_with_seed(shuffle_seed);

            // Resume from position if this is the starting epoch
            if epoch == start_epoch && start_position > 0 {
                train_loader.set_position(start_position);
                if is_main {
                    info!("Resuming from batch position {}", start_position);
                }
            }

            let (train_loss, interrupted) = self.train_epoch_interruptible(train_loader, comm)?;

            // Handle interrupt
            if interrupted {
                if is_main && let Some(dir) = checkpoint_dir {
                    let state = TrainingState {
                        epoch,
                        batch_position: train_loader.position(),
                        global_step: self.global_step,
                        shuffle_seed,
                    };
                    self.save_training_state(dir, &state)?;
                    info!("Saved interrupt checkpoint to {}", dir);
                }
                return Ok(false);
            }

            if is_main {
                info!("Epoch {} train loss: {:.4}", epoch + 1, train_loss);
            }

            if let Some(ref mut eval) = eval_loader {
                let eval_loss = self.evaluate(eval)?;
                if is_main {
                    info!("Epoch {} eval loss: {:.4}", epoch + 1, eval_loss);
                }
            }

            // Only main process saves checkpoints
            if is_main && let Some(dir) = checkpoint_dir {
                let path = format!("{}/checkpoint_epoch_{}.safetensors", dir, epoch + 1);
                self.save_checkpoint(&path)?;
                info!("Saved checkpoint to {}", path);
            }

            // Sync all ranks after checkpoint save
            if let Some(c) = comm {
                c.barrier()?;
            }
        }

        Ok(true)
    }

    /// Train epoch with interrupt checking
    /// Returns (loss, was_interrupted)
    fn train_epoch_interruptible(
        &mut self,
        train_loader: &mut DataLoader,
        comm: Option<&crate::distributed::NcclCommunicator>,
    ) -> Result<(f64, bool)> {
        let is_main = comm.is_none_or(|c| c.rank() == 0);
        let num_batches = train_loader.num_batches();

        let pb = create_progress_bar(num_batches as u64, is_main);

        let mut total_loss = 0.0;
        let mut num_steps = 0;
        let mut accumulated_loss = 0.0;

        while let Some(batch_result) = train_loader.next_batch(&self.device)? {
            // Check for interrupt
            if self.is_interrupted() {
                pb.finish_with_message("interrupted");
                return Ok((total_loss / num_steps.max(1) as f64, true));
            }

            let (input, target) = (batch_result.0, batch_result.1);

            let logits = self.model.forward(&input, 0, true)?;
            let loss = cross_entropy_loss(&logits, &target)?;

            accumulated_loss += loss.to_scalar::<f32>()? as f64;
            num_steps += 1;

            if num_steps % self.training_config.gradient_accumulation_steps == 0 {
                let avg_loss =
                    accumulated_loss / self.training_config.gradient_accumulation_steps as f64;

                self.optimizer.backward_step(&loss)?;

                if let Some(c) = comm {
                    crate::distributed::sync_gradients(&self.var_map, c)?;
                }

                if self.training_config.grad_clip > 0.0 {
                    for var in self.var_map.all_vars() {
                        let grad = var.as_tensor();
                        let norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                        if norm > self.training_config.grad_clip as f32 {
                            let scale = self.training_config.grad_clip as f32 / norm;
                            let _ = var.set(&grad.affine(scale as f64, 0.0)?);
                        }
                    }
                }

                total_loss += avg_loss;
                accumulated_loss = 0.0;
                self.global_step += 1;

                if self
                    .global_step
                    .is_multiple_of(self.training_config.log_every)
                {
                    pb.set_message(format!("{:.4}", avg_loss));
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("done");

        let effective_steps = self.global_step;
        if effective_steps > 0 {
            Ok((total_loss / effective_steps as f64, false))
        } else {
            Ok((0.0, false))
        }
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.var_map.load(path)?;
        Ok(())
    }

    /// Save full training state (weights + progress) for resumable training
    pub fn save_training_state<P: AsRef<Path>>(&self, dir: P, state: &TrainingState) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        // Save model weights
        let weights_path = dir.join("weights.safetensors");
        self.var_map.save(&weights_path)?;

        // Save training state
        let state_path = dir.join("training_state.json");
        let state_json = serde_json::to_string_pretty(state)?;
        std::fs::write(&state_path, state_json)?;

        Ok(())
    }

    /// Load full training state (weights + progress) for resuming
    pub fn load_training_state<P: AsRef<Path>>(&mut self, dir: P) -> Result<TrainingState> {
        let dir = dir.as_ref();

        // Load model weights
        let weights_path = dir.join("weights.safetensors");
        if weights_path.exists() {
            self.var_map.load(&weights_path)?;
        }

        // Load training state
        let state_path = dir.join("training_state.json");
        let state = if state_path.exists() {
            let state_json = std::fs::read_to_string(&state_path)?;
            serde_json::from_str(&state_json)?
        } else {
            TrainingState::default()
        };

        self.global_step = state.global_step;
        Ok(state)
    }

    pub fn model(&self) -> &Transformer {
        &self.model
    }

    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn global_step(&self) -> usize {
        self.global_step
    }
}

// Re-export from generate module for backward compatibility
pub use crate::generate::TextGenerator;
