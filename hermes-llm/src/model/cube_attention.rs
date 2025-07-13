//! CUDA attention built from CubeCL's accelerated Flash Attention and matmuls.

use burn::backend::{
    TensorMetadata,
    ops::{AttentionModuleOptions, FloatTensorOps},
};
use burn::tensor::{FloatDType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::kernel::attention::{AttentionStrategy, attention};
use burn_cubecl::tensor::CubeTensor;
use cubek::attention::forward::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;

use super::cube_tensor::{empty_dtype_like, into_contiguous};
use super::fused_attention::{AttentionBackend, chunked_attention_backward};

const BACKWARD_CHUNK_ROWS: usize = 2048;
const MAX_HEAD_DIM: usize = 128;

fn dimensions(query: &CubeTensor<CudaRuntime>, key: &CubeTensor<CudaRuntime>) -> [usize; 4] {
    let [batch, heads, sequence, head_dim] = query.shape().dims();
    assert_eq!(key.shape().dims(), [batch, heads, sequence, head_dim]);
    assert!(head_dim <= MAX_HEAD_DIM);
    [batch, heads, sequence, head_dim]
}

impl AttentionBackend for CubeBackend<CudaRuntime> {
    fn attention_inner(
        query: CubeTensor<CudaRuntime>,
        key: CubeTensor<CudaRuntime>,
        value: CubeTensor<CudaRuntime>,
        causal: bool,
    ) -> (
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
    ) {
        let output_dtype = query.dtype;
        let query = Self::float_cast(into_contiguous(query), FloatDType::F16);
        let key = Self::float_cast(into_contiguous(key), FloatDType::F16);
        let value = Self::float_cast(into_contiguous(value), FloatDType::F16);
        let [batch, heads, sequence, head_dim] = dimensions(&query, &key);
        assert_eq!(value.shape().dims(), [batch, heads, sequence, head_dim]);

        let options = AttentionModuleOptions {
            is_causal: causal,
            ..Default::default()
        };
        let output = attention(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            None,
            options,
            AttentionStrategy::FlashBlackboxAccelerated(BlackboxAcceleratedStrategy {
                num_planes: 4,
                seq_q: 1,
                seq_kv: 1,
            }),
        )
        .or_else(|_| {
            attention(
                query.clone(),
                key.clone(),
                value.clone(),
                None,
                None,
                options,
                AttentionStrategy::Fallback,
            )
        })
        .expect("Burn attention fallback must support the validated Hermes shape");
        let output_default = Self::float_cast(output.clone(), output_dtype.into());
        let stats = empty_dtype_like(&query, Shape::new([1, 1, 1, 1]), output_dtype);
        (output_default, stats, query, key, value, output)
    }

    fn attention_backward(
        query: CubeTensor<CudaRuntime>,
        key: CubeTensor<CudaRuntime>,
        value: CubeTensor<CudaRuntime>,
        output: CubeTensor<CudaRuntime>,
        _stats: CubeTensor<CudaRuntime>,
        grad_output: CubeTensor<CudaRuntime>,
        causal: bool,
    ) -> (
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
        CubeTensor<CudaRuntime>,
    ) {
        // Forward deliberately saves Q/K/V in FP16. Keep them there for the
        // recompute matmuls instead of expanding to Flex32 and recasting every
        // query chunk. Only elementwise correction needs the output in Flex32.
        let query = into_contiguous(query);
        let key = into_contiguous(key);
        let value = into_contiguous(value);
        let output = Self::float_cast(into_contiguous(output), grad_output.dtype.into());
        let grad_output = into_contiguous(grad_output);
        chunked_attention_backward::<Self>(
            query,
            key,
            value,
            output,
            grad_output,
            causal,
            BACKWARD_CHUNK_ROWS,
        )
    }
}
