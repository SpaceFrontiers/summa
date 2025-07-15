//! CubeCL kernels and dispatch for the fused head + loss.
//!
//! Two row-wise kernels replace a chain of full-vocabulary elementwise
//! passes: an online log-sum-exp statistics pass and a gradient pass with
//! the loss scale and target folded in, emitting the sequence dtype
//! directly. Padded vocabulary columns are never read. The ops cross the
//! lazy-fusion boundary through the same CustomOpIr bridge as the scan.

use burn::backend::TensorMetadata;
use burn::tensor::Shape;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime};

use burn::backend::ops::FloatTensorOps;

use super::{FloatTensor, IntTensor, LinearCrossEntropyBackend};
use crate::model::cube_tensor::{empty_like, empty_like_dtype, into_contiguous};

const CE_THREADS: u32 = 256;

/// Per-row online log-sum-exp statistics plus the target logit: one block
/// owns a row and strides the logical vocabulary once. Padded columns
/// (`col >= logical_vocab`) are never read, which keeps them at exactly
/// zero probability without materializing any mask.
#[allow(clippy::manual_div_ceil, clippy::assign_op_pattern)]
#[cube(launch)]
fn ce_row_statistics(
    logits: &Tensor<f32>,
    bias: &Tensor<f32>,
    targets: &Tensor<i32>,
    stats: &mut Tensor<f32>,
    row_offset: u32,
    stored_vocab: u32,
    logical_vocab: u32,
    #[comptime] use_bias: bool,
) {
    let stored_vocab = stored_vocab as usize;
    let logical_vocab = logical_vocab as usize;
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let threads = CE_THREADS as usize;
    let base = row * stored_vocab;
    let target = usize::cast_from(targets[row_offset as usize + row]);

    let mut running_max = f32::cast_from(f32::NEG_INFINITY);
    let mut running_sum = 0.0f32;
    let mut target_logit = 0.0f32;
    let iterations = (logical_vocab + threads - 1) / threads;
    for i in 0..iterations {
        let col = lane + i * threads;
        if col < logical_vocab {
            let mut x = logits[base + col];
            if use_bias {
                x += bias[col];
            }
            if x > running_max {
                running_sum = running_sum * (running_max - x).exp() + 1.0;
                running_max = x;
            } else {
                running_sum += (x - running_max).exp();
            }
            if col == target {
                target_logit = x;
            }
        }
    }

    let mut shared_max = Shared::new_slice(CE_THREADS as usize);
    let mut shared_sum = Shared::new_slice(CE_THREADS as usize);
    let mut shared_target = Shared::new_slice(CE_THREADS as usize);
    shared_max[lane] = running_max;
    shared_sum[lane] = running_sum;
    shared_target[lane] = target_logit;
    sync_cube();

    #[unroll]
    for level in 0..8 {
        let stride = (CE_THREADS as usize) >> (level + 1);
        if lane < stride {
            let m_a = shared_max[lane];
            let s_a = shared_sum[lane];
            let m_b = shared_max[lane + stride];
            let s_b = shared_sum[lane + stride];
            let m = if m_a > m_b { m_a } else { m_b };
            // Empty lanes carry (-inf, 0); guard the exp so they combine
            // as exact zeros instead of NaN.
            let mut s = 0.0f32;
            if s_a > 0.0 {
                s += s_a * (m_a - m).exp();
            }
            if s_b > 0.0 {
                s += s_b * (m_b - m).exp();
            }
            shared_max[lane] = m;
            shared_sum[lane] = s;
            shared_target[lane] = shared_target[lane] + shared_target[lane + stride];
        }
        sync_cube();
    }

    if lane == 0 {
        stats[row * 3] = shared_max[0];
        stats[row * 3 + 1] = shared_sum[0];
        stats[row * 3 + 2] = shared_target[0];
    }
}

/// Softmax gradient with the target correction and loss scale folded in:
/// `(softmax(x) - onehot(target)) * scale` in a single pass over the
/// chunk. Padded columns receive exact zeros. The gradient is emitted
/// directly in the matmul compute dtype (BF16 on CUDA) — the following
/// tensor-core matmuls consumed a BF16 cast of it anyway, so writing it
/// once removes a full-vocabulary FP32 round trip per chunk.
#[cube(launch)]
fn ce_row_gradient<G: Float>(
    logits: &Tensor<f32>,
    bias: &Tensor<f32>,
    stats: &Tensor<f32>,
    targets: &Tensor<i32>,
    scale: &Tensor<f32>,
    grad: &mut Tensor<G>,
    row_offset: u32,
    stored_vocab: u32,
    logical_vocab: u32,
    #[comptime] use_bias: bool,
) {
    let stored_vocab = stored_vocab as usize;
    let logical_vocab = logical_vocab as usize;
    let idx = ABSOLUTE_POS;
    if idx < grad.len() {
        let row = idx / stored_vocab;
        let col = idx % stored_vocab;
        if col < logical_vocab {
            let mut x = logits[idx];
            if use_bias {
                x += bias[col];
            }
            let m = stats[row * 3];
            let s = stats[row * 3 + 1];
            let step = scale[0];
            let mut value = (x - m).exp() / s * step;
            if col == usize::cast_from(targets[row_offset as usize + row]) {
                value -= step;
            }
            grad[idx] = G::cast_from(value);
        } else {
            grad[idx] = G::cast_from(0.0f32);
        }
    }
}

fn launch_statistics<R: CubeRuntime>(
    logits: &CubeTensor<R>,
    bias: &CubeTensor<R>,
    targets: &CubeTensor<R>,
    row_offset: usize,
    logical_vocab_size: usize,
    use_bias: bool,
) -> CubeTensor<R> {
    let [rows, stored_vocab] = logits.shape().dims();
    let stats = empty_like(logits, Shape::new([rows, 3]));
    ce_row_statistics::launch::<R>(
        &logits.client.clone(),
        CubeCount::Static(rows as u32, 1, 1),
        CubeDim::new_1d(CE_THREADS),
        logits.clone().into_tensor_arg(),
        bias.clone().into_tensor_arg(),
        targets.clone().into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        row_offset as u32,
        stored_vocab as u32,
        logical_vocab_size as u32,
        use_bias,
    );
    stats
}

#[allow(clippy::too_many_arguments)]
fn launch_gradient<R: CubeRuntime>(
    logits: &CubeTensor<R>,
    bias: &CubeTensor<R>,
    stats: &CubeTensor<R>,
    targets: &CubeTensor<R>,
    scale: &CubeTensor<R>,
    row_offset: usize,
    logical_vocab_size: usize,
    use_bias: bool,
    grad_dtype: burn::tensor::DType,
) -> CubeTensor<R> {
    let [rows, stored_vocab] = logits.shape().dims();
    let grad = empty_like_dtype(logits, Shape::new([rows, stored_vocab]), grad_dtype);
    let total = (rows * stored_vocab) as u32;
    macro_rules! launch {
        ($float:ty) => {
            ce_row_gradient::launch::<$float, R>(
                &logits.client.clone(),
                CubeCount::Static(total.div_ceil(CE_THREADS), 1, 1),
                CubeDim::new_1d(CE_THREADS),
                logits.clone().into_tensor_arg(),
                bias.clone().into_tensor_arg(),
                stats.clone().into_tensor_arg(),
                targets.clone().into_tensor_arg(),
                scale.clone().into_tensor_arg(),
                grad.clone().into_tensor_arg(),
                row_offset as u32,
                stored_vocab as u32,
                logical_vocab_size as u32,
                use_bias,
            )
        };
    }
    match grad_dtype {
        burn::tensor::DType::BF16 => launch!(half::bf16),
        burn::tensor::DType::F32 => launch!(f32),
        other => panic!("cross-entropy gradient dtype {other:?} is not supported"),
    }
    grad
}

fn bf16_operand<R: CubeRuntime>(tensor: CubeTensor<R>, enabled: bool) -> CubeTensor<R> {
    if enabled {
        CubeBackend::<R>::float_cast(tensor, burn::tensor::FloatDType::BF16)
    } else {
        tensor
    }
}

/// Chunk matmul mirroring `matmul_2`: BF16 tensor-core operands with an
/// FP32 result on CUDA, plain FP32 elsewhere. `rhs` is pre-cast once by
/// the caller.
fn chunk_matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    bf16: bool,
) -> CubeTensor<R> {
    let out = CubeBackend::<R>::float_matmul(bf16_operand(lhs, bf16), rhs);
    if bf16 {
        CubeBackend::<R>::float_cast(out, burn::tensor::FloatDType::F32)
    } else {
        out
    }
}

/// `(ln(sum) + max - target_logit)` averaged over the chunk, weighted by
/// the chunk length exactly like the reference path.
fn chunk_loss_from_stats<R: CubeRuntime>(stats: CubeTensor<R>, rows: usize) -> CubeTensor<R> {
    type B<R> = CubeBackend<R>;
    let max = B::<R>::float_slice(stats.clone(), &[(0..rows).into(), (0..1).into()]);
    let sum = B::<R>::float_slice(stats.clone(), &[(0..rows).into(), (1..2).into()]);
    let target = B::<R>::float_slice(stats, &[(0..rows).into(), (2..3).into()]);
    let loss = B::<R>::float_sub(B::<R>::float_add(B::<R>::float_log(sum), max), target);
    B::<R>::float_mul_scalar(B::<R>::float_mean(loss), (rows as f32).into())
}

#[allow(clippy::too_many_arguments)]
fn fused_loss<R: CubeRuntime>(
    hidden: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
    targets: CubeTensor<R>,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
    bf16_matmul: bool,
) -> CubeTensor<R> {
    type B<R> = CubeBackend<R>;
    let [tokens, hidden_size] = hidden.shape().dims();
    let bias = into_contiguous(bias);
    let targets = into_contiguous(targets);
    let weight_transposed = bf16_operand(B::<R>::float_swap_dims(weight, 0, 1), bf16_matmul);

    let mut total: Option<CubeTensor<R>> = None;
    for start in (0..tokens).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens);
        let rows = end - start;
        let logits = into_contiguous(chunk_matmul(
            B::<R>::float_slice(
                hidden.clone(),
                &[(start..end).into(), (0..hidden_size).into()],
            ),
            weight_transposed.clone(),
            bf16_matmul,
        ));
        let stats = launch_statistics::<R>(
            &logits,
            &bias,
            &targets,
            start,
            logical_vocab_size,
            use_bias,
        );
        let loss = chunk_loss_from_stats(stats, rows);
        total = Some(match total {
            Some(total) => B::<R>::float_add(total, loss),
            None => loss,
        });
    }
    B::<R>::float_div_scalar(
        total.expect("linear cross-entropy requires at least one token"),
        (tokens as f32).into(),
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_backward<R: CubeRuntime>(
    hidden: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
    targets: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
    bf16_matmul: bool,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    type B<R> = CubeBackend<R>;
    let [tokens, hidden_size] = hidden.shape().dims();
    let [vocab_size, _] = weight.shape().dims();
    let device = hidden.device.clone();
    // The hidden gradient re-enters autodiff where the activation lives,
    // so it must match the incoming activation dtype (BF16 residual
    // stream during training-fusion, FP32 elsewhere) — the fusion IR
    // rejects mismatched gradients at runtime.
    let hidden_dtype = hidden.dtype;
    let bias = into_contiguous(bias);
    let targets = into_contiguous(targets);
    let scale = into_contiguous(B::<R>::float_div_scalar(
        grad_output,
        (tokens as f32).into(),
    ));

    let weight = bf16_operand(weight, bf16_matmul);
    let weight_transposed = B::<R>::float_swap_dims(weight.clone(), 0, 1);
    let mut hidden_gradients = Vec::with_capacity(tokens.div_ceil(chunk_size));
    let mut weight_gradient = B::<R>::float_zeros(
        Shape::new([vocab_size, hidden_size]),
        &device,
        burn::tensor::FloatDType::F32,
    );
    let mut bias_gradient = B::<R>::float_zeros(
        Shape::new([vocab_size]),
        &device,
        burn::tensor::FloatDType::F32,
    );

    for start in (0..tokens).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens);
        let hidden_chunk = B::<R>::float_slice(
            hidden.clone(),
            &[(start..end).into(), (0..hidden_size).into()],
        );
        let logits = into_contiguous(chunk_matmul(
            hidden_chunk.clone(),
            weight_transposed.clone(),
            bf16_matmul,
        ));
        let stats = launch_statistics::<R>(
            &logits,
            &bias,
            &targets,
            start,
            logical_vocab_size,
            use_bias,
        );
        let logits_gradient = launch_gradient::<R>(
            &logits,
            &bias,
            &stats,
            &targets,
            &scale,
            start,
            logical_vocab_size,
            use_bias,
            if bf16_matmul {
                burn::tensor::DType::BF16
            } else {
                burn::tensor::DType::F32
            },
        );
        let logits_gradient_compute = bf16_operand(logits_gradient.clone(), bf16_matmul);
        let hidden_gradient = B::<R>::float_matmul(logits_gradient_compute.clone(), weight.clone());
        hidden_gradients.push(if hidden_gradient.dtype == hidden_dtype {
            hidden_gradient
        } else {
            burn_cubecl::kernel::cast(hidden_gradient, hidden_dtype)
        });
        weight_gradient = B::<R>::float_add(
            weight_gradient,
            chunk_matmul(
                B::<R>::float_swap_dims(logits_gradient_compute, 0, 1),
                bf16_operand(hidden_chunk, bf16_matmul),
                bf16_matmul,
            ),
        );
        if use_bias {
            // The bias gradient accumulates in FP32; sum the BF16 chunk
            // gradient in FP32 so 6k-row column sums keep full precision.
            let gradient_f32 = if logits_gradient.dtype == burn::tensor::DType::F32 {
                logits_gradient
            } else {
                burn_cubecl::kernel::cast(logits_gradient, burn::tensor::DType::F32)
            };
            bias_gradient = B::<R>::float_add(
                bias_gradient,
                B::<R>::float_reshape(
                    B::<R>::float_sum_dim(gradient_f32, 0),
                    Shape::new([vocab_size]),
                ),
            );
        }
    }

    (
        B::<R>::float_cat(hidden_gradients, 0),
        weight_gradient,
        bias_gradient,
    )
}

macro_rules! impl_fused_linear_cross_entropy {
    ($backend:ty, $bf16_matmul:expr) => {
        impl LinearCrossEntropyBackend for $backend {
            fn linear_cross_entropy_inner(
                hidden: FloatTensor<Self>,
                weight: FloatTensor<Self>,
                bias: FloatTensor<Self>,
                targets: IntTensor<Self>,
                logical_vocab_size: usize,
                chunk_size: usize,
                use_bias: bool,
            ) -> FloatTensor<Self> {
                fused_loss(
                    hidden,
                    weight,
                    bias,
                    targets,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                    $bf16_matmul,
                )
            }

            fn linear_cross_entropy_backward(
                hidden: FloatTensor<Self>,
                weight: FloatTensor<Self>,
                bias: FloatTensor<Self>,
                targets: IntTensor<Self>,
                grad_output: FloatTensor<Self>,
                logical_vocab_size: usize,
                chunk_size: usize,
                use_bias: bool,
            ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
                fused_backward(
                    hidden,
                    weight,
                    bias,
                    targets,
                    grad_output,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                    $bf16_matmul,
                )
            }
        }
    };
}

// BF16 tensor-core chunk matmuls on CUDA (mirrors `matmul_2`), FP32 on
// Metal where no BF16 matmul path exists.
#[cfg(feature = "cuda")]
impl_fused_linear_cross_entropy!(CubeBackend<burn_cubecl::cubecl::cuda::CudaRuntime>, true);
#[cfg(feature = "metal")]
impl_fused_linear_cross_entropy!(super::Metal, false);
