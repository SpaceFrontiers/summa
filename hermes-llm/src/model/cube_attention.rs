//! CUDA attention built from CubeCL's accelerated Flash Attention and matmuls.

use burn::backend::{
    TensorMetadata,
    ops::{AttentionModuleOptions, FloatTensorOps},
};
use burn::tensor::FloatDType;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::{cuda::CudaRuntime, prelude::*};
use burn_cubecl::kernel::attention::{AttentionStrategy, attention};
use burn_cubecl::tensor::CubeTensor;
use cubek::attention::forward::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;

use super::cube_tensor::{empty_like, into_contiguous};
use super::fused_attention::AttentionBackend;

const MAX_HEAD_DIM: usize = 128;
const ELEMENTWISE_THREADS: u32 = 256;

#[cube(launch)]
fn causal_mask_kernel(
    scores: &Tensor<f32>,
    output: &mut Tensor<f32>,
    rows: u32,
    sequence: u32,
    row_offset: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < output.len() {
        let column = idx % sequence as usize;
        let row = (idx / sequence as usize) % rows as usize;
        output[idx] = if column > row + row_offset as usize {
            f32::NEG_INFINITY
        } else {
            scores[idx]
        };
    }
}

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
        (output_default, query, key, value, output)
    }

    fn attention_causal_mask(
        scores: CubeTensor<CudaRuntime>,
        row_offset: usize,
    ) -> CubeTensor<CudaRuntime> {
        let scores = into_contiguous(scores);
        let [_, _, rows, sequence] = scores.shape().dims();
        assert!(row_offset + rows <= sequence);
        let total = scores.meta.num_elements();
        let cube_count = CubeCount::Static(
            u32::try_from(total.div_ceil(ELEMENTWISE_THREADS as usize))
                .expect("attention score tensor exceeds the CUDA launch grid"),
            1,
            1,
        );
        let client = scores.client.clone();
        if scores.can_mut() && scores.is_nonoverlapping() {
            causal_mask_kernel::launch::<CudaRuntime>(
                &client,
                cube_count,
                CubeDim::new_1d(ELEMENTWISE_THREADS),
                scores.clone().into_tensor_arg(),
                scores.as_tensor_alias(0),
                rows as u32,
                sequence as u32,
                row_offset as u32,
            );
            scores
        } else {
            let output = empty_like(&scores, scores.shape());
            causal_mask_kernel::launch::<CudaRuntime>(
                &client,
                cube_count,
                CubeDim::new_1d(ELEMENTWISE_THREADS),
                scores.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                rows as u32,
                sequence as u32,
                row_offset as u32,
            );
            output
        }
    }
}
