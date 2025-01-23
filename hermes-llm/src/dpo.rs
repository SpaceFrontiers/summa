use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::io::BufRead;
use std::path::Path;
use tracing::info;

use crate::config::Config;
use crate::model::GPT;
use crate::tokenizer::Tokenizer;

/// A preference pair for DPO training
#[derive(Debug, Deserialize)]
pub struct PreferencePair {
    /// The prompt/context
    pub prompt: String,
    /// The preferred/chosen response
    pub chosen: String,
    /// The rejected response
    pub rejected: String,
}

/// Dataset of preference pairs
pub struct PreferenceDataset {
    pairs: Vec<PreferencePair>,
}

impl PreferenceDataset {
    /// Load preference pairs from a JSONL file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let reader = crate::io::open_file(path)?;
        let mut pairs = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let pair: PreferencePair = serde_json::from_str(&line)?;
            pairs.push(pair);
        }

        info!("Loaded {} preference pairs", pairs.len());
        Ok(Self { pairs })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&PreferencePair> {
        self.pairs.get(idx)
    }

    pub fn iter(&self) -> impl Iterator<Item = &PreferencePair> {
        self.pairs.iter()
    }
}

/// DPO Trainer
pub struct DpoTrainer {
    policy_model: GPT,
    reference_model: GPT,
    optimizer: AdamW,
    var_map: VarMap,
    device: Device,
    beta: f64,
    max_len: usize,
}

impl DpoTrainer {
    pub fn new(
        config: Config,
        checkpoint_path: &str,
        device: Device,
        lr: f64,
        beta: f64,
        max_len: usize,
    ) -> Result<Self> {
        // Create policy model (will be trained)
        let mut var_map = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let policy_model = GPT::new(&config, vb)?;
        var_map.load(checkpoint_path)?;

        // Create reference model (frozen copy)
        let mut ref_var_map = VarMap::new();
        let ref_vb = candle_nn::VarBuilder::from_varmap(&ref_var_map, DType::F32, &device);
        let reference_model = GPT::new(&config, ref_vb)?;
        ref_var_map.load(checkpoint_path)?;

        let params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.0,
            eps: 1e-8,
        };
        let optimizer = AdamW::new(var_map.all_vars(), params)?;

        info!(
            "DPO Trainer initialized: beta={}, max_len={}, lr={}",
            beta, max_len, lr
        );

        Ok(Self {
            policy_model,
            reference_model,
            optimizer,
            var_map,
            device,
            beta,
            max_len,
        })
    }

    /// Compute log probabilities for a sequence
    fn compute_log_probs(
        &self,
        model: &GPT,
        input_ids: &Tensor,
        target_ids: &Tensor,
    ) -> Result<Tensor> {
        let logits = model.forward(input_ids, 0, false)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, 2)?;

        // Gather log probs for target tokens
        let (batch_size, seq_len, vocab_size) = log_probs.dims3()?;
        let target_ids_flat = target_ids.flatten_all()?;

        // Reshape for gather: [batch * seq, vocab]
        let log_probs_2d = log_probs.reshape((batch_size * seq_len, vocab_size))?;

        // Create indices for gather
        let indices = target_ids_flat.unsqueeze(1)?;
        let gathered = log_probs_2d.gather(&indices, 1)?;

        // Reshape back and sum over sequence
        let seq_log_probs = gathered.reshape((batch_size, seq_len))?;
        let total_log_prob = seq_log_probs.sum(1)?;

        Ok(total_log_prob)
    }

    /// Compute DPO loss for a batch
    /// DPO Loss = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_ref(y_w|x) + log_ref(y_l|x))))
    fn compute_dpo_loss(&self, chosen_ids: &Tensor, rejected_ids: &Tensor) -> Result<Tensor> {
        // Compute log probs for policy model
        let policy_chosen_logp = self.compute_log_probs(
            &self.policy_model,
            &chosen_ids.narrow(1, 0, chosen_ids.dim(1)? - 1)?,
            &chosen_ids.narrow(1, 1, chosen_ids.dim(1)? - 1)?,
        )?;

        let policy_rejected_logp = self.compute_log_probs(
            &self.policy_model,
            &rejected_ids.narrow(1, 0, rejected_ids.dim(1)? - 1)?,
            &rejected_ids.narrow(1, 1, rejected_ids.dim(1)? - 1)?,
        )?;

        // Compute log probs for reference model (no grad)
        let ref_chosen_logp = self.compute_log_probs(
            &self.reference_model,
            &chosen_ids.narrow(1, 0, chosen_ids.dim(1)? - 1)?,
            &chosen_ids.narrow(1, 1, chosen_ids.dim(1)? - 1)?,
        )?;

        let ref_rejected_logp = self.compute_log_probs(
            &self.reference_model,
            &rejected_ids.narrow(1, 0, rejected_ids.dim(1)? - 1)?,
            &rejected_ids.narrow(1, 1, rejected_ids.dim(1)? - 1)?,
        )?;

        // Compute rewards
        let chosen_rewards = (&policy_chosen_logp - &ref_chosen_logp)?;
        let rejected_rewards = (&policy_rejected_logp - &ref_rejected_logp)?;

        // DPO loss: -log(sigmoid(beta * (r_w - r_l)))
        let logits = ((&chosen_rewards - &rejected_rewards)? * self.beta)?;

        // -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        let neg_logits = logits.neg()?;
        let loss = neg_logits.broadcast_add(&Tensor::new(1.0f32, &self.device)?)?;
        let loss = (loss.exp()? + 1.0)?.log()?;
        let loss = loss.mean_all()?;

        Ok(loss)
    }

    /// Train for one epoch
    pub fn train_epoch(
        &mut self,
        dataset: &PreferenceDataset,
        tokenizer: &Tokenizer,
        batch_size: usize,
    ) -> Result<f64> {
        let num_batches = dataset.len().div_ceil(batch_size);
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} loss: {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        let mut total_loss = 0.0;
        let mut num_steps = 0;

        for batch_start in (0..dataset.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(dataset.len());

            // Tokenize batch
            let mut chosen_ids_batch = Vec::new();
            let mut rejected_ids_batch = Vec::new();

            for i in batch_start..batch_end {
                let pair = dataset.get(i).unwrap();

                let chosen_text = format!("{}{}", pair.prompt, pair.chosen);
                let rejected_text = format!("{}{}", pair.prompt, pair.rejected);

                let mut chosen_ids = tokenizer.encode(&chosen_text, false)?;
                let mut rejected_ids = tokenizer.encode(&rejected_text, false)?;

                // Truncate/pad to max_len
                chosen_ids.truncate(self.max_len);
                rejected_ids.truncate(self.max_len);

                while chosen_ids.len() < self.max_len {
                    chosen_ids.push(tokenizer.pad_token_id());
                }
                while rejected_ids.len() < self.max_len {
                    rejected_ids.push(tokenizer.pad_token_id());
                }

                chosen_ids_batch.push(chosen_ids);
                rejected_ids_batch.push(rejected_ids);
            }

            // Convert to tensors
            let chosen_tensor = Tensor::new(
                chosen_ids_batch
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<u32>>(),
                &self.device,
            )?
            .reshape((chosen_ids_batch.len(), self.max_len))?;

            let rejected_tensor = Tensor::new(
                rejected_ids_batch
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<u32>>(),
                &self.device,
            )?
            .reshape((rejected_ids_batch.len(), self.max_len))?;

            // Compute loss
            let loss = self.compute_dpo_loss(&chosen_tensor, &rejected_tensor)?;
            let loss_val = loss.to_scalar::<f32>()? as f64;

            // Backward pass
            self.optimizer.backward_step(&loss)?;

            total_loss += loss_val;
            num_steps += 1;

            pb.set_message(format!("{:.4}", loss_val));
            pb.inc(1);
        }

        pb.finish();

        Ok(total_loss / num_steps as f64)
    }

    /// Full training loop
    pub fn train(
        &mut self,
        dataset: &PreferenceDataset,
        tokenizer: &Tokenizer,
        epochs: usize,
        batch_size: usize,
        output_dir: Option<&str>,
    ) -> Result<()> {
        for epoch in 0..epochs {
            info!("Epoch {}/{}", epoch + 1, epochs);
            let avg_loss = self.train_epoch(dataset, tokenizer, batch_size)?;
            info!("Epoch {} complete, avg loss: {:.4}", epoch + 1, avg_loss);

            if let Some(dir) = output_dir {
                std::fs::create_dir_all(dir)?;
                let path = format!("{}/dpo_epoch_{}.safetensors", dir, epoch + 1);
                self.var_map.save(&path)?;
                info!("Saved checkpoint to {}", path);
            }
        }

        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.var_map.save(path)?;
        Ok(())
    }
}
