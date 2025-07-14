//! CUDA attention built from CubeCL's accelerated Flash Attention and matmuls.

use burn::backend::{
    TensorMetadata,
    ops::{AttentionModuleOptions, FloatTensorOps},
};
use burn::tensor::FloatDType;
use burn::tensor::Shape;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::{cuda::CudaRuntime, prelude::*};
use burn_cubecl::kernel::attention::{AttentionStrategy, attention};
use burn_cubecl::tensor::CubeTensor;
use cubek::attention::forward::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;
use half::bf16;

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

const SOFTMAX_THREADS: u32 = 256;

/// Per-row online softmax statistics over the causally-visible prefix. The
/// causal bound replaces any materialized mask, and the softmax scale is
/// folded into the pass.
#[allow(clippy::manual_div_ceil)]
#[cube(launch)]
fn attention_softmax_stats(
    scores: &Tensor<f32>,
    stats: &mut Tensor<f32>,
    cols: u32,
    chunk_rows: u32,
    row_offset: u32,
    scale: f32,
    #[comptime] causal: bool,
) {
    let cols = cols as usize;
    let chunk_rows = chunk_rows as usize;
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let threads = SOFTMAX_THREADS as usize;
    let mut bound = cols;
    if causal {
        let visible = row % chunk_rows + row_offset as usize + 1;
        if visible < cols {
            bound = visible;
        }
    }
    let base = row * cols;

    let mut running_max = f32::cast_from(f32::NEG_INFINITY);
    let mut running_sum = 0.0f32;
    let iterations = (bound + threads - 1) / threads;
    for i in 0..iterations {
        let col = lane + i * threads;
        if col < bound {
            let x = scores[base + col] * scale;
            if x > running_max {
                running_sum = running_sum * (running_max - x).exp() + 1.0;
                running_max = x;
            } else {
                running_sum += (x - running_max).exp();
            }
        }
    }

    let mut shared_max = Shared::new_slice(SOFTMAX_THREADS as usize);
    let mut shared_sum = Shared::new_slice(SOFTMAX_THREADS as usize);
    shared_max[lane] = running_max;
    shared_sum[lane] = running_sum;
    sync_cube();

    #[unroll]
    for level in 0..8 {
        let stride = (SOFTMAX_THREADS as usize) >> (level + 1);
        if lane < stride {
            let m_a = shared_max[lane];
            let s_a = shared_sum[lane];
            let m_b = shared_max[lane + stride];
            let s_b = shared_sum[lane + stride];
            let m = if m_a > m_b { m_a } else { m_b };
            // Lanes past the causal bound carry (-inf, 0); guard the exp so
            // they combine as exact zeros instead of NaN.
            let mut sum = 0.0f32;
            if s_a > 0.0 {
                sum += s_a * (m_a - m).exp();
            }
            if s_b > 0.0 {
                sum += s_b * (m_b - m).exp();
            }
            shared_max[lane] = m;
            shared_sum[lane] = sum;
        }
        sync_cube();
    }

    if lane == 0 {
        stats[row * 2] = shared_max[0];
        stats[row * 2 + 1] = shared_sum[0];
    }
}

/// Emits softmax probabilities and score gradients for the backward chunk in
/// one pass, both in BF16 for the following tensor-core matmuls. Positions
/// past the causal bound receive exact zeros.
#[cube(launch)]
fn attention_backward_probabilities_kernel(
    scores: &Tensor<f32>,
    stats: &Tensor<f32>,
    grad_probabilities: &Tensor<f32>,
    correction: &Tensor<f32>,
    probabilities: &mut Tensor<bf16>,
    score_gradient: &mut Tensor<bf16>,
    total: u32,
    cols: u32,
    chunk_rows: u32,
    row_offset: u32,
    scale: f32,
    #[comptime] causal: bool,
) {
    let cols = cols as usize;
    let chunk_rows = chunk_rows as usize;
    let idx = ABSOLUTE_POS;
    if idx < total as usize {
        let row = idx / cols;
        let col = idx % cols;
        let mut bound = cols;
        if causal {
            let visible = row % chunk_rows + row_offset as usize + 1;
            if visible < cols {
                bound = visible;
            }
        }
        if col < bound {
            let m = stats[row * 2];
            let s = stats[row * 2 + 1];
            let p = (scores[idx] * scale - m).exp() / s;
            let ds = p * (grad_probabilities[idx] - correction[row]) * scale;
            probabilities[idx] = bf16::cast_from(p);
            score_gradient[idx] = bf16::cast_from(ds);
        } else {
            probabilities[idx] = bf16::cast_from(0.0f32);
            score_gradient[idx] = bf16::cast_from(0.0f32);
        }
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

    fn attention_backward_probabilities(
        scores: CubeTensor<CudaRuntime>,
        grad_probabilities: CubeTensor<CudaRuntime>,
        correction: CubeTensor<CudaRuntime>,
        scale: f32,
        row_offset: usize,
        causal: bool,
    ) -> (CubeTensor<CudaRuntime>, CubeTensor<CudaRuntime>) {
        let scores = into_contiguous(scores);
        let grad_probabilities = into_contiguous(grad_probabilities);
        let correction = into_contiguous(correction);
        // The kernels read these buffers as raw FP32; a mismatched dtype is
        // reinterpreted bit-for-bit (NaN garbage), never an error downstream.
        for (name, tensor) in [
            ("scores", &scores),
            ("probability gradient", &grad_probabilities),
            ("correction", &correction),
        ] {
            assert_eq!(
                tensor.dtype,
                burn::tensor::DType::F32,
                "attention backward {name} must be FP32"
            );
        }
        let [batch, heads, chunk_rows, cols] = scores.shape().dims();
        let rows = batch * heads * chunk_rows;
        let client = scores.client.clone();
        let device = scores.device.clone();
        let stats = empty_like(&scores, Shape::new([rows, 2]));
        attention_softmax_stats::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(SOFTMAX_THREADS),
            scores.clone().into_tensor_arg(),
            stats.clone().into_tensor_arg(),
            cols as u32,
            chunk_rows as u32,
            row_offset as u32,
            scale,
            causal,
        );
        let shape = Shape::new([batch, heads, chunk_rows, cols]);
        let probabilities = Self::float_empty(shape.clone(), &device, FloatDType::BF16);
        let score_gradient = Self::float_empty(shape, &device, FloatDType::BF16);
        let total = (rows * cols) as u32;
        attention_backward_probabilities_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(total.div_ceil(SOFTMAX_THREADS), 1, 1),
            CubeDim::new_1d(SOFTMAX_THREADS),
            scores.into_tensor_arg(),
            stats.into_tensor_arg(),
            grad_probabilities.into_tensor_arg(),
            correction.into_tensor_arg(),
            probabilities.clone().into_tensor_arg(),
            score_gradient.clone().into_tensor_arg(),
            total,
            cols as u32,
            chunk_rows as u32,
            row_offset as u32,
            scale,
            causal,
        );
        (probabilities, score_gradient)
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
