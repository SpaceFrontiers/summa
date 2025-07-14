//! Memory-bounded output projection and cross-entropy for training.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
use burn::backend::{
    Backend, Dispatch, DispatchKindConversion, DispatchTensor, NdArray, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{IndexingUpdateOp, Int};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_nn::loss::CrossEntropyLossConfig;

use super::matmul::{matmul_2, matmul_input};

/// Backend capability for training without retaining full-vocabulary logits.
#[backend_extension(
    Cuda: cfg(feature = "cuda"),
    Metal: cfg(feature = "metal"),
    NdArray,
    Autodiff,
)]
pub trait LinearCrossEntropyBackend: Backend {
    fn linear_cross_entropy_inner(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: IntTensor<Self>,
        logical_vocab_size: usize,
        chunk_size: usize,
        use_bias: bool,
    ) -> FloatTensor<Self>;

    #[allow(unused_variables, clippy::too_many_arguments)]
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
        panic!("linear cross-entropy only supports first-order autodiff")
    }
}

pub(super) fn linear_cross_entropy(
    hidden: Tensor<2>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    targets: Tensor<1, Int>,
    logical_vocab_size: usize,
    chunk_size: usize,
) -> Tensor<1> {
    assert!(
        chunk_size > 0,
        "linear cross-entropy chunk size must be positive"
    );
    let use_bias = bias.is_some();
    let bias = bias.unwrap_or_else(|| Tensor::zeros([weight.dims()[0]], &hidden.device()));
    let output = Dispatch::linear_cross_entropy_inner(
        hidden.into_dispatch(),
        weight.into_dispatch(),
        bias.into_dispatch(),
        targets.into_dispatch(),
        logical_vocab_size,
        chunk_size,
        use_bias,
    );
    Tensor::from_dispatch(output)
}

fn mask_padded_logits(logits: Tensor<2>, logical_vocab_size: usize) -> Tensor<2> {
    let [tokens, stored_vocab_size] = logits.dims();
    assert!(
        logical_vocab_size > 0 && logical_vocab_size <= stored_vocab_size,
        "logical vocabulary {logical_vocab_size} must be in 1..={stored_vocab_size}"
    );
    if logical_vocab_size == stored_vocab_size {
        return logits;
    }

    let device = logits.device();
    logits.slice_assign(
        [0..tokens, logical_vocab_size..stored_vocab_size],
        Tensor::full(
            [tokens, stored_vocab_size - logical_vocab_size],
            f32::NEG_INFINITY,
            &device,
        ),
    )
}

fn chunked_loss<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
) -> FloatTensor<B>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let hidden = Tensor::<2>::from_primitive::<B>(hidden);
    let weight = Tensor::<2>::from_primitive::<B>(weight);
    let bias = Tensor::<1>::from_primitive::<B>(bias);
    let targets = Tensor::<1, Int>::from_primitive::<B>(targets);
    let [tokens, hidden_size] = hidden.dims();
    let [vocab_size, weight_hidden] = weight.dims();
    assert_eq!(hidden_size, weight_hidden);
    assert_eq!(targets.dims(), [tokens]);
    assert_eq!(bias.dims(), [vocab_size]);

    let criterion = CrossEntropyLossConfig::new().init(&hidden.device());
    let weight_transposed = matmul_input(weight.transpose());
    let mut total = None;
    for start in (0..tokens).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens);
        let logits = matmul_2(
            hidden.clone().slice([start..end, 0..hidden_size]),
            weight_transposed.clone(),
        );
        let logits = if use_bias {
            logits + bias.clone().reshape([1, vocab_size])
        } else {
            logits
        };
        let logits = mask_padded_logits(logits, logical_vocab_size);
        let loss = criterion.forward(logits, targets.clone().slice(start..end));
        let loss = loss.mul_scalar((end - start) as f32);
        total = Some(match total {
            Some(total) => total + loss,
            None => loss,
        });
    }
    total
        .expect("linear cross-entropy requires at least one token")
        .div_scalar(tokens as f32)
        .try_into_primitive::<B>()
        .expect("linear cross-entropy output stayed on its input backend")
}

#[allow(clippy::too_many_arguments)]
fn chunked_backward<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    grad_output: FloatTensor<B>,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let hidden = Tensor::<2>::from_primitive::<B>(hidden);
    let weight = Tensor::<2>::from_primitive::<B>(weight);
    let bias = Tensor::<1>::from_primitive::<B>(bias);
    let targets = Tensor::<1, Int>::from_primitive::<B>(targets);
    let grad_output = Tensor::<1>::from_primitive::<B>(grad_output).reshape([1, 1]);
    let [tokens, hidden_size] = hidden.dims();
    let [vocab_size, weight_hidden] = weight.dims();
    assert_eq!(hidden_size, weight_hidden);
    let device = hidden.device();
    let scale = grad_output.div_scalar(tokens as f32);
    let weight = matmul_input(weight);
    let weight_transposed = weight.clone().transpose();
    let mut hidden_gradients = Vec::with_capacity(tokens.div_ceil(chunk_size));
    let mut weight_gradient = Tensor::<2>::zeros([vocab_size, hidden_size], &device);
    let mut bias_gradient = Tensor::<1>::zeros([vocab_size], &device);

    for start in (0..tokens).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens);
        let chunk_tokens = end - start;
        let hidden_chunk = hidden.clone().slice([start..end, 0..hidden_size]);
        let logits = matmul_2(hidden_chunk.clone(), weight_transposed.clone());
        let logits = if use_bias {
            logits + bias.clone().reshape([1, vocab_size])
        } else {
            logits
        };
        let logits = mask_padded_logits(logits, logical_vocab_size);
        let target_indices = targets.clone().slice(start..end).reshape([chunk_tokens, 1]);
        let corrections = Tensor::<2>::ones([chunk_tokens, 1], &device).neg();
        let logits_gradient =
            softmax(logits, 1).scatter(1, target_indices, corrections, IndexingUpdateOp::Add)
                * scale.clone();
        let logits_gradient_compute = matmul_input(logits_gradient.clone());
        hidden_gradients.push(matmul_2(logits_gradient_compute.clone(), weight.clone()));
        weight_gradient =
            weight_gradient + matmul_2(logits_gradient_compute.transpose(), hidden_chunk);
        if use_bias {
            bias_gradient = bias_gradient + logits_gradient.sum_dim(0).reshape([vocab_size]);
        }
    }

    (
        Tensor::cat(hidden_gradients, 0)
            .try_into_primitive::<B>()
            .expect("hidden gradient stayed on its input backend"),
        weight_gradient
            .try_into_primitive::<B>()
            .expect("weight gradient stayed on its input backend"),
        bias_gradient
            .try_into_primitive::<B>()
            .expect("bias gradient stayed on its input backend"),
    )
}

macro_rules! impl_reference_linear_cross_entropy {
    ($backend:ty) => {
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
                chunked_loss::<Self>(
                    hidden,
                    weight,
                    bias,
                    targets,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
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
                chunked_backward::<Self>(
                    hidden,
                    weight,
                    bias,
                    targets,
                    grad_output,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                )
            }
        }
    };
}

impl_reference_linear_cross_entropy!(burn_ndarray::NdArray);

/// GPU fast path: the per-chunk softmax/NLL math runs as two row-wise fused
/// kernels instead of a chain of full-vocabulary tensor passes. The reference
/// path above needs ~18 full-logit passes per chunk (mask, bias, softmax,
/// scatter, scale); the fused version needs three (statistics, loss/grad,
/// plus the producing matmul), which is what makes chunked cross-entropy
/// bandwidth-comparable to a fused framework implementation.
#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
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
            let hidden_gradient =
                B::<R>::float_matmul(logits_gradient_compute.clone(), weight.clone());
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
}

#[derive(Clone, Debug)]
struct LinearCrossEntropyState<B: LinearCrossEntropyBackend> {
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
}

#[derive(Debug)]
struct LinearCrossEntropyBackward;

impl<B: LinearCrossEntropyBackend> Backward<B, 3> for LinearCrossEntropyBackward {
    type State = LinearCrossEntropyState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_hidden, node_weight, node_bias] = ops.parents;
        let grad_output = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let gradients = B::linear_cross_entropy_backward(
            state.hidden,
            state.weight,
            state.bias,
            state.targets,
            grad_output,
            state.logical_vocab_size,
            state.chunk_size,
            state.use_bias,
        );
        for (node, gradient) in [
            (node_hidden, gradients.0),
            (node_weight, gradients.1),
            (node_bias, gradients.2),
        ] {
            if let Some(node) = node {
                grads.register::<B>(node.id, gradient);
            }
        }
    }
}

impl<B: LinearCrossEntropyBackend, C: CheckpointStrategy> LinearCrossEntropyBackend
    for Autodiff<B, C>
{
    fn linear_cross_entropy_inner(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: IntTensor<Self>,
        logical_vocab_size: usize,
        chunk_size: usize,
        use_bias: bool,
    ) -> FloatTensor<Self> {
        match LinearCrossEntropyBackward
            .prepare::<C>([hidden.node.clone(), weight.node.clone(), bias.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let targets = <Self as burn::backend::AutodiffBackend>::int_inner(targets);
                let output = B::linear_cross_entropy_inner(
                    hidden.primitive.clone(),
                    weight.primitive.clone(),
                    bias.primitive.clone(),
                    targets.clone(),
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                );
                let state = LinearCrossEntropyState {
                    hidden: hidden.primitive,
                    weight: weight.primitive,
                    bias: bias.primitive,
                    targets,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                };
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                let targets = <Self as burn::backend::AutodiffBackend>::int_inner(targets);
                let output = B::linear_cross_entropy_inner(
                    hidden.primitive,
                    weight.primitive,
                    bias.primitive,
                    targets,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                );
                prep.finish(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    fn max_diff(lhs: TensorData, rhs: TensorData) -> f32 {
        lhs.convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(rhs.convert::<f32>().to_vec::<f32>().unwrap())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn chunked_loss_and_gradients_match_materialized_cross_entropy() {
        let device = Device::ndarray().autodiff();
        let (tokens, hidden_size, logical_vocab_size) = (7, 5, 11);
        let hidden_data = (0..tokens * hidden_size)
            .map(|index| (index as f32 * 0.071).sin() * 0.2)
            .collect::<Vec<_>>();
        let target_data = vec![0_i64, 3, 7, 2, 10, 4, 1];

        for stored_vocab_size in [logical_vocab_size, 16] {
            let weight_data = (0..stored_vocab_size * hidden_size)
                .map(|index| (index as f32 * 0.113).cos() * 0.3)
                .collect::<Vec<_>>();
            let bias_data = (0..stored_vocab_size)
                .map(|index| index as f32 * 0.01)
                .collect::<Vec<_>>();

            let run = |chunked: bool, use_bias: bool| {
                let hidden = Tensor::<2>::from_data(
                    TensorData::new(hidden_data.clone(), [tokens, hidden_size]),
                    &device,
                )
                .require_grad();
                let weight = Tensor::<2>::from_data(
                    TensorData::new(weight_data.clone(), [stored_vocab_size, hidden_size]),
                    &device,
                )
                .require_grad();
                let bias = Tensor::<1>::from_data(
                    TensorData::new(bias_data.clone(), [stored_vocab_size]),
                    &device,
                )
                .require_grad();
                let targets = Tensor::<1, Int>::from_data(
                    TensorData::new(target_data.clone(), [tokens]),
                    &device,
                );
                let loss = if chunked {
                    linear_cross_entropy(
                        hidden.clone(),
                        weight.clone(),
                        use_bias.then(|| bias.clone()),
                        targets,
                        logical_vocab_size,
                        3,
                    )
                } else {
                    let weight = weight
                        .clone()
                        .slice([0..logical_vocab_size, 0..hidden_size]);
                    let logits = hidden.clone().matmul(weight.transpose());
                    let logits = if use_bias {
                        logits
                            + bias
                                .clone()
                                .slice(0..logical_vocab_size)
                                .reshape([1, logical_vocab_size])
                    } else {
                        logits
                    };
                    CrossEntropyLossConfig::new()
                        .init(&device)
                        .forward(logits, targets)
                };
                let loss_data = loss.clone().into_data();
                let mut gradients = loss.backward();
                (
                    loss_data,
                    hidden.grad_remove(&mut gradients).unwrap().into_data(),
                    weight.grad_remove(&mut gradients).unwrap().into_data(),
                    bias.grad_remove(&mut gradients).map(Tensor::into_data),
                )
            };

            for use_bias in [false, true] {
                let expected = run(false, use_bias);
                let actual = run(true, use_bias);
                if stored_vocab_size > logical_vocab_size {
                    let padded_weight_gradient = actual
                        .2
                        .clone()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .skip(logical_vocab_size * hidden_size);
                    assert!(padded_weight_gradient.into_iter().all(|value| value == 0.0));
                    if let Some(bias_gradient) = &actual.3 {
                        let padded_bias_gradient = bias_gradient
                            .clone()
                            .convert::<f32>()
                            .to_vec::<f32>()
                            .unwrap()
                            .into_iter()
                            .skip(logical_vocab_size);
                        assert!(padded_bias_gradient.into_iter().all(|value| value == 0.0));
                    }
                }
                assert!(max_diff(expected.0, actual.0) < 1e-6);
                assert!(max_diff(expected.1, actual.1) < 1e-6);
                assert!(max_diff(expected.2, actual.2) < 1e-6);
                match (expected.3, actual.3) {
                    (Some(expected), Some(actual)) => assert!(max_diff(expected, actual) < 1e-6),
                    (None, None) => {}
                    _ => panic!("bias gradient tracking differs"),
                }
            }
        }
    }

    /// GPU fused kernels vs the NdArray reference: padding below the reduction
    /// width (empty lanes), bias on/off, and a chunk boundary offset.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    #[test]
    fn gpu_fused_loss_and_gradients_match_cpu_reference() {
        fn gpu_device() -> Device {
            #[cfg(feature = "metal")]
            return Device::metal(burn::tensor::DeviceKind::DefaultDevice);

            #[cfg(all(feature = "cuda", not(feature = "metal")))]
            return Device::cuda(0);
        }

        let (tokens, hidden_size, stored_vocab, logical_vocab, chunk) = (7, 8, 64, 50, 3);
        let hidden_data: Vec<f32> = (0..tokens * hidden_size)
            .map(|i| ((i * 37 + 11) % 23) as f32 * 0.11 - 1.2)
            .collect();
        let weight_data: Vec<f32> = (0..stored_vocab * hidden_size)
            .map(|i| ((i * 29 + 5) % 19) as f32 * 0.07 - 0.6)
            .collect();
        let bias_data: Vec<f32> = (0..stored_vocab)
            .map(|i| ((i * 13 + 3) % 17) as f32 * 0.05 - 0.4)
            .collect();
        let target_data: Vec<i64> = (0..tokens)
            .map(|i| ((i * 7 + 2) % logical_vocab) as i64)
            .collect();

        for use_bias in [false, true] {
            let mut outputs = Vec::new();
            for device in [Device::ndarray().autodiff(), gpu_device().autodiff()] {
                let hidden = Tensor::<2>::from_data(
                    TensorData::new(hidden_data.clone(), [tokens, hidden_size]),
                    &device,
                )
                .require_grad();
                let weight = Tensor::<2>::from_data(
                    TensorData::new(weight_data.clone(), [stored_vocab, hidden_size]),
                    &device,
                )
                .require_grad();
                let bias = Tensor::<1>::from_data(
                    TensorData::new(bias_data.clone(), [stored_vocab]),
                    &device,
                )
                .require_grad();
                let targets = Tensor::<1, Int>::from_data(
                    TensorData::new(target_data.clone(), [tokens]),
                    &device,
                );
                let loss = linear_cross_entropy(
                    hidden.clone(),
                    weight.clone(),
                    use_bias.then(|| bias.clone()),
                    targets,
                    logical_vocab,
                    chunk,
                );
                let grads = loss.backward();
                outputs.push((
                    loss.into_data(),
                    hidden.grad(&grads).unwrap().into_data(),
                    weight.grad(&grads).unwrap().into_data(),
                    bias.grad(&grads).map(|grad| grad.into_data()),
                ));
            }
            let gpu = outputs.pop().unwrap();
            let cpu = outputs.pop().unwrap();
            let loss_diff = max_diff(cpu.0.clone(), gpu.0.clone());
            let hidden_diff = max_diff(cpu.1.clone(), gpu.1.clone());
            let weight_diff = max_diff(cpu.2.clone(), gpu.2.clone());
            eprintln!(
                "use_bias={use_bias} loss_diff={loss_diff:.3e} hidden_diff={hidden_diff:.3e} weight_diff={weight_diff:.3e}"
            );
            // CUDA runs the chunk matmuls on BF16 tensor cores (like real
            // training), so every comparison against the f32 CPU reference
            // carries BF16 rounding; Metal computes in f32 and lands ~1e-7.
            assert!(loss_diff < 1e-2);
            // Gradients flow through BF16 tensor-core matmuls on CUDA, so the
            // comparison against the f32 CPU reference uses the same 1e-2
            // bound as the wider training parity checks.
            assert!(max_diff(cpu.1, gpu.1) < 1e-2);
            assert!(max_diff(cpu.2.clone(), gpu.2.clone()) < 1e-2);
            match (cpu.3, gpu.3) {
                (Some(cpu_bias), Some(gpu_bias)) => assert!(max_diff(cpu_bias, gpu_bias) < 1e-2),
                (None, None) => {}
                _ => panic!("bias gradient presence diverged between devices"),
            }
            // Padded rows must receive exactly zero gradient on the GPU path.
            let padded = gpu.2.convert::<f32>().to_vec::<f32>().unwrap()
                [logical_vocab * hidden_size..]
                .to_vec();
            assert!(padded.into_iter().all(|value| value == 0.0));
        }
    }
}
