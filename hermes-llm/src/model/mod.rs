//! Shared language-model stack for training and inference.
//!
//! One [`Transformer`](transformer::Transformer) implementation serves
//! training (`hermes-train`), generation, and retrieval — there is no
//! second model or checkpoint adapter. The module tree is layered:
//!
//! - **Model** — [`transformer`], [`block`], [`attention`], [`mamba`],
//!   [`ffn`], [`norm`]: architecture built from Burn modules.
//! - **Kernel families** — [`scan`], [`linear_cross_entropy`], [`conv`],
//!   plus [`fused_attention`]/[`cube_attention`]: each exposes a backend
//!   extension trait, a differentiable tensor-op reference (the
//!   correctness oracle), an autodiff node, and CubeCL GPU kernels, with
//!   CPU-vs-GPU parity tests per family. Size limits and measured tuning
//!   are catalogued in `docs/kernel-tuning-surface.md`.
//! - **Runtime glue** — [`backend`] (device selection), [`matmul`]
//!   (precision-policy matmul entry points), [`cube_tensor`] (raw-tensor
//!   helpers), and [`fusion`] (CustomOpIr bridges that let the custom ops
//!   cross Burn's lazy-fusion boundary under `training-fusion`).

mod attention;
pub mod backend;
mod block;
mod conv;
#[cfg(feature = "cuda")]
mod cube_attention;
#[cfg(any(feature = "metal", feature = "cuda"))]
mod cube_tensor;
mod ffn;
mod fused_attention;
#[cfg(feature = "cuda")]
mod fused_swiglu;
#[cfg(feature = "training-fusion")]
mod fusion;
#[cfg(feature = "cuda")]
mod grouped_linear;
mod linear_cross_entropy;
mod mamba;
mod matmul;
#[cfg(feature = "cuda")]
mod moe_dispatch;
#[cfg(feature = "cuda")]
mod moe_route;
mod moe_topk;
mod norm;
mod row_permute;
mod scan;
mod transformer;
mod weights;

pub use attention::{AttnCache, MultiHeadAttention};
pub use backend::{Device, default_device};
pub use block::{InferenceState, LayerState, TransformerBlock};
pub use ffn::FeedForward;
pub use fused_attention::AttentionBackend;
pub use linear_cross_entropy::LinearCrossEntropyBackend;
pub use mamba::{MambaMixer, MambaState};
pub use norm::Norm;
pub use scan::MambaBackend;
pub use transformer::Transformer;
pub use weights::{load_safetensors, save_safetensors};

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod test_support {
    use burn::tensor::{Tensor, TensorData};

    pub(super) fn values(len: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * scale).sin() * 0.25 + offset)
            .collect()
    }

    pub(super) fn max_diff(lhs: TensorData, rhs: TensorData) -> f32 {
        lhs.convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(rhs.convert::<f32>().to_vec::<f32>().unwrap())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    pub(super) fn snapshot<const D: usize>(tensor: Tensor<D>) -> TensorData {
        let shape = tensor.shape();
        TensorData::new(
            tensor.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
            shape,
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::module::Module;
    use burn::prelude::*;
    use burn::tensor::{Int, TensorData};
    use burn_nn::RotaryEncodingConfig;

    use super::*;
    use crate::mal::{ModelDef, NormType, PositionEncoding, get_builtin_model};

    fn hybrid_test_config() -> ModelDef {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 3;
        config.max_seq_len = 16;
        if let Some(pattern) = config.pattern.as_mut() {
            for block in pattern {
                block.ffn.hidden_dim = Some(16);
                block.attention.num_heads = Some(2);
                block.attention.num_kv_heads = Some(1);
                block.attention.head_dim = Some(4);
            }
        }
        config
    }

    fn max_abs_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
        let a = a.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let b = b.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        a.into_iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_norm_parameters_and_identity() {
        let device = Device::ndarray();
        let layer_norm = Norm::new(NormType::LayerNorm, 8, 1e-5, &device);
        let identity = Norm::new(NormType::None, 8, 1e-5, &device);

        assert_eq!(layer_norm.num_params(), 16);
        assert_eq!(identity.num_params(), 0);
    }

    #[test]
    fn test_attention_cached_matches_stateless() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.hidden_size = 16;
        config.max_seq_len = 16;
        config.block.attention.num_heads = Some(4);
        config.block.attention.num_kv_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        let block = config.block.clone();
        let device = Device::ndarray();
        device.seed(7);

        let attention = MultiHeadAttention::new(&config, &block, &device);
        let rope = RotaryEncodingConfig::new(16, 4).init(&device);
        let data: Vec<f32> = (0..6 * config.hidden_size)
            .map(|i| (i as f32 * 0.071).sin())
            .collect();
        let x = Tensor::from_data(TensorData::new(data, [1, 6, config.hidden_size]), &device);

        let full = attention.forward(x.clone(), &rope, 0);
        let mut cache = attention.make_cache(1, &device);
        let prefill = attention.forward_cached(
            x.clone().slice([0..1, 0..4, 0..config.hidden_size]),
            &rope,
            0,
            &mut cache,
        );
        let decode = attention.forward_cached(
            x.slice([0..1, 4..6, 0..config.hidden_size]),
            &rope,
            4,
            &mut cache,
        );

        assert!(max_abs_diff(prefill, full.clone().slice([0..1, 0..4, 0..16])) < 1e-5);
        assert!(max_abs_diff(decode, full.slice([0..1, 4..6, 0..16])) < 1e-5);
        assert_eq!(cache.len(), 6);
    }

    #[test]
    fn test_attention_respects_disabled_position_encoding() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.hidden_size = 16;
        config.block.attention.num_heads = Some(4);
        config.block.attention.num_kv_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        config.block.attention.position_encoding = PositionEncoding::None;
        config.block.attention.causal = false;
        let device = Device::ndarray();
        device.seed(9);
        let attention = MultiHeadAttention::new(&config, &config.block, &device);
        let rope = RotaryEncodingConfig::new(16, 4).init(&device);
        let data: Vec<f32> = (0..4 * config.hidden_size)
            .map(|i| (i as f32 * 0.097).sin())
            .collect();
        let x = Tensor::from_data(TensorData::new(data, [1, 4, config.hidden_size]), &device);

        let at_zero = attention.forward(x.clone(), &rope, 0);
        let at_offset = attention.forward(x, &rope, 5);
        assert!(max_abs_diff(at_zero, at_offset) < 1e-6);
    }

    #[test]
    fn test_attention_stateless_offset_preserves_relative_mask() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.hidden_size = 16;
        config.max_seq_len = 16;
        config.block.attention.num_heads = Some(4);
        config.block.attention.num_kv_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        config.block.attention.position_encoding = PositionEncoding::None;
        config.block.attention.causal = true;
        config.block.attention.window_size = Some(2);
        let device = Device::ndarray();
        device.seed(10);
        let attention = MultiHeadAttention::new(&config, &config.block, &device);
        let rope = RotaryEncodingConfig::new(16, 4).init(&device);
        let data: Vec<f32> = (0..4 * config.hidden_size)
            .map(|i| (i as f32 * 0.089).sin())
            .collect();
        let x = Tensor::from_data(TensorData::new(data, [1, 4, config.hidden_size]), &device);

        let at_zero = attention.forward(x.clone(), &rope, 0);
        let at_offset = attention.forward(x, &rope, 5);
        assert!(max_abs_diff(at_zero, at_offset) < 1e-6);
    }

    #[test]
    fn test_mamba_stateful_matches_stateless() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.hidden_size = 8;
        let block = config.block_for_layer(0).clone();
        let ssm = block.ssm.as_ref().unwrap();
        let device = Device::ndarray();
        device.seed(11);

        let mixer = MambaMixer::new(&config, ssm, &device);
        let data: Vec<f32> = (0..6 * config.hidden_size)
            .map(|i| (i as f32 * 0.113).cos())
            .collect();
        let x = Tensor::from_data(TensorData::new(data, [1, 6, config.hidden_size]), &device);

        let full = mixer.forward(x.clone());
        let mut state = mixer.make_state(1, &device);
        let prefill = mixer.forward_with_state(
            x.clone().slice([0..1, 0..4, 0..config.hidden_size]),
            Some(&mut state),
        );
        let decode = mixer.forward_with_state(
            x.slice([0..1, 4..6, 0..config.hidden_size]),
            Some(&mut state),
        );

        assert!(max_abs_diff(prefill, full.clone().slice([0..1, 0..4, 0..8])) < 1e-5);
        assert!(max_abs_diff(decode, full.slice([0..1, 4..6, 0..8])) < 1e-5);
        assert_eq!(state.conv.dims(), [1, 16, 3]);
        assert_eq!(state.h.dims(), [1, 16, 16]);
    }

    #[test]
    fn test_hybrid_transformer_stateful_matches_stateless() {
        let config = hybrid_test_config();
        let device = Device::ndarray();
        device.seed(19);
        let model = Transformer::new(&config, &device).unwrap();
        let ids = vec![1_i64, 7, 3, 9, 2, 5];
        let input = Tensor::<2, Int>::from_data(TensorData::new(ids.clone(), [1, 6]), &device);

        let full = model.forward(input, 0);
        assert_eq!(full.dims(), [1, 6, 32]);
        assert_eq!(model.config().vocab_size, 32);
        assert_eq!(model.config().padded_vocab_size(), 64);

        let mut state = model.make_state_with_capacity(1, ids.len(), &device);
        assert_eq!(state.capacity(), ids.len());
        let prefill = model.forward_with_state(
            Tensor::<2, Int>::from_data(TensorData::new(ids[..4].to_vec(), [1, 4]), &device),
            &mut state,
        );
        let decode = model.forward_with_state(
            Tensor::<2, Int>::from_data(TensorData::new(ids[4..].to_vec(), [1, 2]), &device),
            &mut state,
        );

        assert!(max_abs_diff(prefill, full.clone().slice([0..1, 0..4, 0..32])) < 1e-4);
        assert!(max_abs_diff(decode, full.slice([0..1, 4..6, 0..32])) < 1e-4);
        assert_eq!(state.pos(), 6);
    }

    #[test]
    fn test_transformer_rejects_zero_sized_ssm_dimensions() {
        let device = Device::ndarray();
        let mut cases = Vec::new();

        let mut config = hybrid_test_config();
        config.pattern.as_mut().unwrap()[0]
            .ssm
            .as_mut()
            .unwrap()
            .expand = 0;
        cases.push(("expand", config));

        let mut config = hybrid_test_config();
        config.pattern.as_mut().unwrap()[0]
            .ssm
            .as_mut()
            .unwrap()
            .state_dim = 0;
        cases.push(("state_dim", config));

        let mut config = hybrid_test_config();
        config.pattern.as_mut().unwrap()[0]
            .ssm
            .as_mut()
            .unwrap()
            .conv_kernel = 0;
        cases.push(("conv_kernel", config));

        let mut config = hybrid_test_config();
        config.pattern.as_mut().unwrap()[0]
            .ssm
            .as_mut()
            .unwrap()
            .dt_rank = Some(0);
        cases.push(("dt_rank", config));

        for (field, config) in cases {
            let err = Transformer::new(&config, &device)
                .err()
                .unwrap_or_else(|| panic!("zero {field} was accepted"));
            assert!(err.to_string().contains(field), "{field}: {err}");
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn test_cuda_prepared_inference_matches_ephemeral_weight_casts() {
        let config = hybrid_test_config();
        let device = default_device();
        device.seed(29);
        let mut model = Transformer::new(&config, &device).unwrap();
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(vec![1_i64, 7, 3, 9, 2, 5], [1, 6]),
            &device,
        );
        let before = model.forward(input.clone(), 0).into_data();
        model.prepare_inference();
        let after = model.forward(input, 0).into_data();
        let difference = before
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(after.convert::<f32>().to_vec::<f32>().unwrap())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max);
        assert!(difference < 1e-3, "maximum logit difference: {difference}");
    }

    #[test]
    fn test_tied_embeddings_remove_lm_head_parameters() {
        let mut untied = get_builtin_model("tiny").unwrap();
        untied.vocab_size = 32;
        untied.hidden_size = 8;
        untied.num_layers = 1;
        untied.block.ffn.hidden_dim = Some(16);
        untied.block.attention.num_heads = Some(2);
        untied.block.attention.head_dim = Some(4);
        let mut tied = untied.clone();
        tied.embeddings.tie_weights = true;
        let device = Device::ndarray();
        device.seed(23);

        let untied = Transformer::new(&untied, &device).unwrap();
        let tied = Transformer::new(&tied, &device).unwrap();

        assert_eq!(untied.num_parameters() - tied.num_parameters(), 64 * 8);
    }

    #[test]
    fn test_output_bias_is_counted_for_tied_and_untied_heads() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 1;
        config.block.ffn.hidden_dim = Some(16);
        config.block.attention.num_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        let device = Device::ndarray();

        for tied in [false, true] {
            config.embeddings.tie_weights = tied;
            config.output.bias = false;
            let without = Transformer::new(&config, &device).unwrap().num_parameters();
            config.output.bias = true;
            let with = Transformer::new(&config, &device).unwrap().num_parameters();
            assert_eq!(with - without, config.padded_vocab_size());
        }
    }
}
