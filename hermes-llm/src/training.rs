use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::info;

use crate::config::{Config, TrainingConfig};
use crate::data::DataLoader;
use crate::model::{GPT, cross_entropy_loss};

pub struct Trainer {
    model: GPT,
    optimizer: AdamW,
    var_map: VarMap,
    #[allow(dead_code)]
    config: Config,
    training_config: TrainingConfig,
    device: Device,
    global_step: usize,
}

impl Trainer {
    pub fn new(config: Config, training_config: TrainingConfig, device: Device) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = GPT::new(&config, vb)?;

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

        Ok(Self {
            model,
            optimizer,
            var_map,
            config,
            training_config,
            device,
            global_step: 0,
        })
    }

    pub fn train_epoch(&mut self, train_loader: &mut DataLoader) -> Result<f64> {
        self.train_epoch_distributed(train_loader, None)
    }

    pub fn train_epoch_distributed(
        &mut self,
        train_loader: &mut DataLoader,
        comm: Option<&crate::distributed::NcclCommunicator>,
    ) -> Result<f64> {
        let num_batches = train_loader.num_batches();
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} loss: {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        let mut total_loss = 0.0;
        let mut num_steps = 0;
        let mut accumulated_loss = 0.0;

        train_loader.reset();

        while let Some(batch_result) = train_loader.next_batch(&self.device)? {
            let (input, target) = (batch_result.0, batch_result.1);

            let logits = self.model.forward(&input, 0, true)?;
            let loss = cross_entropy_loss(&logits, &target)?;

            accumulated_loss += loss.to_scalar::<f32>()? as f64;
            num_steps += 1;

            if num_steps % self.training_config.gradient_accumulation_steps == 0 {
                let avg_loss =
                    accumulated_loss / self.training_config.gradient_accumulation_steps as f64;

                self.optimizer.backward_step(&loss)?;

                // Synchronize gradients across all ranks (all-reduce average)
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
            Ok(total_loss / effective_steps as f64)
        } else {
            Ok(0.0)
        }
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
    ) -> Result<()> {
        self.train_distributed(train_loader, eval_loader, checkpoint_dir, None)
    }

    pub fn train_distributed(
        &mut self,
        train_loader: &mut DataLoader,
        mut eval_loader: Option<&mut DataLoader>,
        checkpoint_dir: Option<&str>,
        comm: Option<&crate::distributed::NcclCommunicator>,
    ) -> Result<()> {
        let _is_distributed = comm.is_some();
        let is_main = comm.is_none_or(|c| c.rank() == 0);

        info!(
            "Starting training for {} epochs",
            self.training_config.epochs
        );

        for epoch in 0..self.training_config.epochs {
            info!("Epoch {}/{}", epoch + 1, self.training_config.epochs);

            let train_loss = self.train_epoch_distributed(train_loader, comm)?;
            info!("Epoch {} train loss: {:.4}", epoch + 1, train_loss);

            if let Some(ref mut eval) = eval_loader {
                let eval_loss = self.evaluate(eval)?;
                info!("Epoch {} eval loss: {:.4}", epoch + 1, eval_loss);
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

        Ok(())
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.var_map.load(path)?;
        Ok(())
    }

    pub fn model(&self) -> &GPT {
        &self.model
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn global_step(&self) -> usize {
        self.global_step
    }
}

pub fn get_lr_with_warmup(
    step: usize,
    warmup_steps: usize,
    max_lr: f64,
    min_lr: f64,
    total_steps: usize,
) -> f64 {
    if step < warmup_steps {
        max_lr * (step as f64 / warmup_steps as f64)
    } else {
        let decay_ratio = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
        let coeff = 0.5 * (1.0 + (std::f64::consts::PI * decay_ratio).cos());
        min_lr + coeff * (max_lr - min_lr)
    }
}

pub struct TextGenerator<'a> {
    model: &'a GPT,
    device: &'a Device,
}

impl<'a> TextGenerator<'a> {
    pub fn new(model: &'a GPT, device: &'a Device) -> Self {
        Self { model, device }
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
    ) -> Result<Vec<u32>> {
        use rand::Rng;

        let mut tokens = prompt_tokens.to_vec();
        let mut rng = rand::rng();

        for _ in 0..max_new_tokens {
            let context_len = tokens.len().min(self.model.config().max_seq_len);
            let context: Vec<u32> = tokens[tokens.len() - context_len..].to_vec();

            let input = Tensor::new(context.as_slice(), self.device)?
                .unsqueeze(0)?
                .to_dtype(DType::U32)?;

            let logits = self.model.forward(&input, 0, false)?;
            let logits = logits.narrow(1, context_len - 1, 1)?.squeeze(1)?;

            let logits = if temperature != 1.0 {
                logits.affine(1.0 / temperature, 0.0)?
            } else {
                logits
            };

            let logits = if let Some(k) = top_k {
                let logits_vec: Vec<f32> = logits.to_vec1()?;
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut masked = vec![f32::NEG_INFINITY; logits_vec.len()];
                for i in 0..k.min(indexed.len()) {
                    masked[indexed[i].0] = indexed[i].1;
                }
                Tensor::new(masked, self.device)?
            } else {
                logits
            };

            let probs = candle_nn::ops::softmax_last_dim(&logits)?;
            let probs_vec: Vec<f32> = probs.to_vec1()?;

            let cumsum: Vec<f32> = probs_vec
                .iter()
                .scan(0.0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect();

            let r: f32 = rng.random();
            let next_token = cumsum.iter().position(|&p| p > r).unwrap_or(0) as u32;

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
