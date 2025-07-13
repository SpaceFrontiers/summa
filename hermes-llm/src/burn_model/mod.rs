//! Burn inference stack. GPU tensor operations use Burn's CubeCL backend; the
//! custom Mamba selective scan runs directly on the same CubeCL tensors.

mod attention;
pub mod backend;
mod block;
mod conv;
mod ffn;
mod mamba;
mod norm;
mod scan;
mod transformer;
mod weights;

pub use attention::{AttnCache, MultiHeadAttention};
pub use backend::{Backend, Device, default_device};
pub use block::{InferenceState, LayerState, TransformerBlock};
pub use ffn::FeedForward;
pub use mamba::{MambaMixer, MambaState};
pub use norm::Norm;
pub use scan::MambaBackend;
pub use transformer::Transformer;
pub use weights::{load_safetensors, save_safetensors};

#[cfg(test)]
mod tests {
    use burn::module::Module;
    use burn::prelude::*;
    use burn::tensor::{Int, TensorData};
    use burn_ndarray::NdArray;
    use burn_nn::RotaryEncodingConfig;

    use super::*;
    use crate::mal::{NormType, PositionEncoding, get_builtin_model};

    type TestBackend = NdArray<f32>;

    fn max_abs_diff<const D: usize>(a: Tensor<TestBackend, D>, b: Tensor<TestBackend, D>) -> f32 {
        let a = a.into_data().to_vec::<f32>().unwrap();
        let b = b.into_data().to_vec::<f32>().unwrap();
        a.into_iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_burn_norm_parameters_and_identity() {
        let device = Default::default();
        let layer_norm = Norm::<TestBackend>::new(NormType::LayerNorm, 8, 1e-5, &device);
        let identity = Norm::<TestBackend>::new(NormType::None, 8, 1e-5, &device);

        assert_eq!(layer_norm.num_params(), 16);
        assert_eq!(identity.num_params(), 0);
    }

    #[test]
    fn test_burn_attention_cached_matches_stateless() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.hidden_size = 16;
        config.max_seq_len = 16;
        config.block.attention.num_heads = Some(4);
        config.block.attention.num_kv_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        let block = config.block.clone();
        let device = Default::default();
        TestBackend::seed(&device, 7);

        let attention = MultiHeadAttention::<TestBackend>::new(&config, &block, &device);
        let rope = RotaryEncodingConfig::new(16, 4).init(&device);
        let data: Vec<f32> = (0..6 * config.hidden_size)
            .map(|i| (i as f32 * 0.071).sin())
            .collect();
        let x = Tensor::from_data(TensorData::new(data, [1, 6, config.hidden_size]), &device);

        let full = attention.forward(x.clone(), &rope, 0);
        let mut cache = AttnCache::default();
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
        assert_eq!(cache.k.unwrap().dims(), [1, 2, 6, 4]);
        assert_eq!(cache.v.unwrap().dims(), [1, 2, 6, 4]);
    }

    #[test]
    fn test_burn_attention_respects_disabled_position_encoding() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.hidden_size = 16;
        config.block.attention.num_heads = Some(4);
        config.block.attention.num_kv_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        config.block.attention.position_encoding = PositionEncoding::None;
        config.block.attention.causal = false;
        let device = Default::default();
        TestBackend::seed(&device, 9);
        let attention = MultiHeadAttention::<TestBackend>::new(&config, &config.block, &device);
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
    fn test_burn_mamba_stateful_matches_stateless() {
        let mut config = get_builtin_model("hybrid-tiny").unwrap();
        config.hidden_size = 8;
        let block = config.block_for_layer(0).clone();
        let ssm = block.ssm.as_ref().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 11);

        let mixer = MambaMixer::<TestBackend>::new(&config, ssm, &device);
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
    fn test_burn_hybrid_transformer_stateful_matches_stateless() {
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
        let device = Default::default();
        TestBackend::seed(&device, 19);
        let model = Transformer::<TestBackend>::new(&config, &device).unwrap();
        let ids = vec![1_i64, 7, 3, 9, 2, 5];
        let input =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(ids.clone(), [1, 6]), &device);

        let full = model.forward(input, 0);
        assert_eq!(full.dims(), [1, 6, 32]);

        let mut state = model.make_state(1, &device);
        let prefill = model.forward_with_state(
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(ids[..4].to_vec(), [1, 4]),
                &device,
            ),
            &mut state,
        );
        let decode = model.forward_with_state(
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(ids[4..].to_vec(), [1, 2]),
                &device,
            ),
            &mut state,
        );

        assert!(max_abs_diff(prefill, full.clone().slice([0..1, 0..4, 0..32])) < 1e-4);
        assert!(max_abs_diff(decode, full.slice([0..1, 4..6, 0..32])) < 1e-4);
        assert_eq!(state.pos(), 6);
    }

    #[test]
    fn test_burn_tied_embeddings_remove_lm_head_parameters() {
        let mut untied = get_builtin_model("tiny").unwrap();
        untied.vocab_size = 32;
        untied.hidden_size = 8;
        untied.num_layers = 1;
        untied.block.ffn.hidden_dim = Some(16);
        untied.block.attention.num_heads = Some(2);
        untied.block.attention.head_dim = Some(4);
        let mut tied = untied.clone();
        tied.embeddings.tie_weights = true;
        let device = Default::default();
        TestBackend::seed(&device, 23);

        let untied = Transformer::<TestBackend>::new(&untied, &device).unwrap();
        let tied = Transformer::<TestBackend>::new(&tied, &device).unwrap();

        assert_eq!(untied.num_parameters() - tied.num_parameters(), 32 * 8);
    }

    #[test]
    fn test_burn_output_bias_is_counted_for_tied_and_untied_heads() {
        let mut config = get_builtin_model("tiny").unwrap();
        config.vocab_size = 32;
        config.hidden_size = 8;
        config.num_layers = 1;
        config.block.ffn.hidden_dim = Some(16);
        config.block.attention.num_heads = Some(2);
        config.block.attention.head_dim = Some(4);
        let device = Default::default();

        for tied in [false, true] {
            config.embeddings.tie_weights = tied;
            config.output.bias = false;
            let without = Transformer::<TestBackend>::new(&config, &device)
                .unwrap()
                .num_parameters();
            config.output.bias = true;
            let with = Transformer::<TestBackend>::new(&config, &device)
                .unwrap()
                .num_parameters();
            assert_eq!(with - without, config.vocab_size);
        }
    }
}
