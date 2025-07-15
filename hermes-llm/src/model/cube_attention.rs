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
use cubek::attention::forward::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions,
};
use cubek::attention::forward::launch as cubek_launch;
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

/// Emits softmax probabilities and score gradients for the backward chunk in
/// one pass, both in BF16 for the following tensor-core matmuls. Positions
/// past the causal bound receive exact zeros.
#[cube(launch)]
fn attention_backward_probabilities_kernel(
    scores: &Tensor<f32>,
    log_sum_exp: &Tensor<f32>,
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
            // Square attention: the LSE row stride equals the column count.
            let lse_index = (row / chunk_rows) * cols + row_offset as usize + row % chunk_rows;
            let p = (scores[idx] * scale - log_sum_exp[lse_index]).exp();
            let ds = p * (grad_probabilities[idx] - correction[row]) * scale;
            probabilities[idx] = bf16::cast_from(p);
            score_gradient[idx] = bf16::cast_from(ds);
        } else {
            probabilities[idx] = bf16::cast_from(0.0f32);
            score_gradient[idx] = bf16::cast_from(0.0f32);
        }
    }
}

/// Log-sum-exp for the tensor-op fallback path: materializes the scaled,
/// causally-masked score matrix and reduces it row-wise. Only runs at the
/// small shapes the flash kernel rejects.
fn fallback_log_sum_exp(
    query: &CubeTensor<CudaRuntime>,
    key: &CubeTensor<CudaRuntime>,
    causal: bool,
) -> CubeTensor<CudaRuntime> {
    type B = CubeBackend<CudaRuntime>;
    let [batch, heads, sequence, head_dim] = query.shape().dims();
    let query = B::float_cast(query.clone(), FloatDType::F32);
    let key = B::float_cast(key.clone(), FloatDType::F32);
    let scores = B::float_matmul(query, B::float_swap_dims(key, 2, 3));
    let scores = B::float_div_scalar(scores, ((head_dim as f32).sqrt()).into());
    let scores = if causal {
        <B as AttentionBackend>::attention_causal_mask(scores, 0)
    } else {
        scores
    };
    let max = B::float_max_dim(scores.clone(), 3);
    let shifted = B::float_exp(B::float_sub(scores, max.clone()));
    let sum = B::float_sum_dim(shifted, 3);
    let log_sum_exp = B::float_add(B::float_log(sum), max);
    B::float_reshape(log_sum_exp, Shape::new([batch * heads, sequence]))
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
        CubeTensor<CudaRuntime>,
    ) {
        let output_dtype = query.dtype;
        let query = Self::float_cast(into_contiguous(query), FloatDType::F16);
        let key = Self::float_cast(into_contiguous(key), FloatDType::F16);
        let value = Self::float_cast(into_contiguous(value), FloatDType::F16);
        let [batch, heads, sequence, head_dim] = dimensions(&query, &key);
        assert_eq!(value.shape().dims(), [batch, heads, sequence, head_dim]);

        // Flash forward through cubek directly so the kernel also emits the
        // per-row log-sum-exp the backward needs; burn's module op has no
        // LSE surface.
        let device = query.device.clone();
        let output = empty_like(&query, Shape::new([batch, heads, sequence, head_dim]));
        let log_sum_exp = Self::float_empty(
            Shape::new([batch * heads, sequence]),
            &device,
            FloatDType::F32,
        );
        let dtypes = AttentionGlobalTypes {
            query: burn::backend::cubecl::dtype_to_storage_type(query.dtype),
            key: burn::backend::cubecl::dtype_to_storage_type(key.dtype),
            value: burn::backend::cubecl::dtype_to_storage_type(value.dtype),
            mask: burn::backend::cubecl::dtype_to_storage_type(burn::tensor::DType::U8),
            out: burn::backend::cubecl::dtype_to_storage_type(output.dtype),
        };
        let flash = cubek_launch::launch_ref_with_lse::<CudaRuntime>(
            cubek_launch::Strategy::BlackboxAccelerated(cubek_launch::BlueprintStrategy::Inferred(
                BlackboxAcceleratedStrategy {
                    num_planes: 4,
                    seq_q: 1,
                    seq_kv: 1,
                },
            )),
            &query.client.clone(),
            query.clone().binding(),
            key.clone().binding(),
            value.clone().binding(),
            None,
            output.clone().binding(),
            log_sum_exp.clone().binding(),
            &dtypes,
            AttentionOptions {
                causal,
                accumulator_precision: AccumulatorPrecision::Strict(
                    burn_cubecl::cubecl::ir::StorageType::Scalar(
                        burn_cubecl::cubecl::ir::ElemType::Float(
                            burn_cubecl::cubecl::ir::FloatKind::F32,
                        ),
                    ),
                ),
            },
        );

        let (output, log_sum_exp) = match flash {
            Ok(()) => (output, log_sum_exp),
            Err(_) => {
                // Shapes the flash kernel rejects take burn's tensor-op
                // fallback; the log-sum-exp is then recomputed from a
                // materialized score matrix (test-scale shapes only).
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
                    AttentionStrategy::Fallback,
                )
                .expect("Burn attention fallback must support the validated Hermes shape");
                let log_sum_exp = fallback_log_sum_exp(&query, &key, causal);
                (output, log_sum_exp)
            }
        };

        let output_default = Self::float_cast(output.clone(), output_dtype.into());
        (output_default, query, key, value, output, log_sum_exp)
    }

    fn attention_backward_probabilities(
        scores: CubeTensor<CudaRuntime>,
        grad_probabilities: CubeTensor<CudaRuntime>,
        correction: CubeTensor<CudaRuntime>,
        log_sum_exp: CubeTensor<CudaRuntime>,
        scale: f32,
        row_offset: usize,
        causal: bool,
    ) -> (CubeTensor<CudaRuntime>, CubeTensor<CudaRuntime>) {
        let scores = into_contiguous(scores);
        let grad_probabilities = into_contiguous(grad_probabilities);
        let correction = into_contiguous(correction);
        let log_sum_exp = into_contiguous(log_sum_exp);
        // The kernels read these buffers as raw FP32; a mismatched dtype is
        // reinterpreted bit-for-bit (NaN garbage), never an error downstream.
        for (name, tensor) in [
            ("scores", &scores),
            ("probability gradient", &grad_probabilities),
            ("correction", &correction),
            ("log-sum-exp", &log_sum_exp),
        ] {
            assert_eq!(
                tensor.dtype,
                burn::tensor::DType::F32,
                "attention backward {name} must be FP32"
            );
        }
        let [batch, heads, chunk_rows, cols] = scores.shape().dims();
        assert_eq!(
            log_sum_exp.shape().dims(),
            [batch * heads, cols],
            "attention backward LSE must be [batch * heads, seq] for square attention"
        );
        let rows = batch * heads * chunk_rows;
        let client = scores.client.clone();
        let device = scores.device.clone();
        let shape = Shape::new([batch, heads, chunk_rows, cols]);
        let probabilities = Self::float_empty(shape.clone(), &device, FloatDType::BF16);
        let score_gradient = Self::float_empty(shape, &device, FloatDType::BF16);
        let total = (rows * cols) as u32;
        attention_backward_probabilities_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(total.div_ceil(SOFTMAX_THREADS), 1, 1),
            CubeDim::new_1d(SOFTMAX_THREADS),
            scores.into_tensor_arg(),
            log_sum_exp.into_tensor_arg(),
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
