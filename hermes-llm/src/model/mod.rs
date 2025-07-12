mod attention;
mod block;
mod ffn;
mod mamba;
mod norm;
mod rope;
mod transformer;

pub use attention::{AttnCache, MultiHeadAttention};
pub use block::{InferenceState, LayerState, Mixer, TransformerBlock};
pub use ffn::FeedForward;
pub use mamba::{MambaMixer, MambaState};
pub use norm::{LayerNorm, Norm, RMSNorm};
pub use rope::RotaryEmbedding;
pub use transformer::{Transformer, cross_entropy_loss};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mal::{ModelDef, get_builtin_model};
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn forward_random(config: &ModelDef) -> Result<Tensor> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(config, vb)?;
        let ids = Tensor::zeros((2, 12), DType::U32, &device)?;
        model.forward(&ids, 0, false)
    }

    #[test]
    fn test_hybrid_forward() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        let logits = forward_random(&config).unwrap();
        assert_eq!(logits.dims3().unwrap(), (2, 12, 128));
        let sum = logits
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(sum.is_finite());
    }

    #[test]
    fn test_hybrid_tensor_names() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let _model = Transformer::new(&config, vb).unwrap();

        let names: std::collections::HashSet<String> =
            var_map.data().lock().unwrap().keys().cloned().collect();

        // Pattern [ssm, ssm, attn] over 6 layers: layers 0,1,3,4 are SSM
        for i in [0usize, 1, 3, 4] {
            for t in [
                "in_proj.weight",
                "conv1d.weight",
                "conv1d.bias",
                "x_proj.weight",
                "dt_proj.weight",
                "dt_proj.bias",
                "A_log",
                "D",
                "out_proj.weight",
            ] {
                assert!(
                    names.contains(&format!("layers.{i}.ssm.{t}")),
                    "missing layers.{i}.ssm.{t}"
                );
            }
            assert!(!names.contains(&format!("layers.{i}.attention.q_proj.weight")));
        }
        // Layers 2, 5 are attention
        for i in [2usize, 5] {
            assert!(names.contains(&format!("layers.{i}.attention.q_proj.weight")));
            assert!(!names.contains(&format!("layers.{i}.ssm.in_proj.weight")));
        }
    }

    #[test]
    fn test_qk_norm_tensors_and_stateful_parity() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        config.block.attention.qk_norm = true;
        if let Some(pattern) = config.pattern.as_mut() {
            for b in pattern.iter_mut() {
                b.attention.qk_norm = true;
            }
        }

        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb).unwrap();

        let names: std::collections::HashSet<String> =
            var_map.data().lock().unwrap().keys().cloned().collect();
        // Attention layers (2, 5 in the [ssm,ssm,attn] pattern) get q/k norms
        assert!(names.contains("layers.2.attention.q_norm.weight"));
        assert!(names.contains("layers.5.attention.k_norm.weight"));
        assert!(!names.contains("layers.0.attention.q_norm.weight")); // ssm layer

        // Stateful decode still matches stateless with qk-norm active
        let ids: Vec<u32> = (0..10).map(|i| (i * 5 + 1) % 128).collect();
        let full = Tensor::new(ids.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_logits = model.forward(&full, 0, false).unwrap();

        let mut state = model.make_state(1, &device).unwrap();
        let prefill = Tensor::new(&ids[..9], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        model.forward_with_state(&prefill, &mut state).unwrap();
        let step = Tensor::new(&ids[9..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let step_logits = model.forward_with_state(&step, &mut state).unwrap();

        let a: Vec<f32> = step_logits.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = full_logits
            .narrow(1, 9, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "qk-norm stateful diverges: {max_diff}");
    }

    #[test]
    fn test_tied_embeddings() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        config.embeddings.tie_weights = true;

        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb).unwrap();

        // No lm_head tensor when tied
        let names: Vec<String> = var_map.data().lock().unwrap().keys().cloned().collect();
        assert!(!names.iter().any(|n| n.starts_with("lm_head")));

        let ids = Tensor::zeros((1, 6), DType::U32, &device).unwrap();
        let logits = model.forward(&ids, 0, false).unwrap();
        assert_eq!(logits.dims3().unwrap(), (1, 6, 128));
    }

    #[test]
    fn test_stateful_matches_stateless() {
        // Prefill + single-token decode must reproduce full-recompute logits
        // for both mixer types (hybrid covers attention KV cache and Mamba
        // recurrent state).
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 128;
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = Transformer::new(&config, vb).unwrap();

        let ids: Vec<u32> = (0..12).map(|i| (i * 7 + 3) % 128).collect();
        let full_input = Tensor::new(ids.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let full_logits = model.forward(&full_input, 0, false).unwrap(); // [1, 12, V]

        // Prefill 8 tokens, then decode 4 one at a time
        let mut state = model.make_state(1, &device).unwrap();
        let prefill = Tensor::new(&ids[..8], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mut logits = model.forward_with_state(&prefill, &mut state).unwrap();
        // Last prefill row == stateless row 7
        let check = |step_logits: &Tensor, pos: usize| {
            let a: Vec<f32> = step_logits
                .narrow(1, step_logits.dim(1).unwrap() - 1, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = full_logits
                .narrow(1, pos, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let max_diff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "logits diverge at pos {pos}: {max_diff}");
        };
        check(&logits, 7);

        for (step, &tok) in ids[8..].iter().enumerate() {
            let input = Tensor::new(&[tok], &device).unwrap().unsqueeze(0).unwrap();
            logits = model.forward_with_state(&input, &mut state).unwrap();
            check(&logits, 8 + step);
        }
        assert_eq!(state.pos(), 12);
    }

    #[test]
    fn test_legacy_transformer_forward_unchanged() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 64;
        let logits = forward_random(&config).unwrap();
        assert_eq!(logits.dims3().unwrap(), (2, 12, 64));
    }
}
