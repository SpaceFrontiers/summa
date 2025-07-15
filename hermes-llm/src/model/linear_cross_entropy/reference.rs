//! Portable chunked implementation of the fused head + loss.
//!
//! Processes the sequence in fixed row chunks so the `[rows, vocab]`
//! logits are never materialized whole; padded vocabulary columns are
//! masked to `-inf` so their probability (and gradient) is exactly zero.
//! Runs on any backend and serves as the oracle for the GPU kernels.

use burn::backend::{
    Backend, DispatchKindConversion, DispatchTensor,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{IndexingUpdateOp, Int};
use burn_nn::loss::CrossEntropyLossConfig;

use crate::model::matmul::{matmul_2, matmul_input};

use super::LinearCrossEntropyBackend;

pub(super) fn mask_padded_logits(logits: Tensor<2>, logical_vocab_size: usize) -> Tensor<2> {
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

pub(super) fn chunked_loss<B: Backend>(
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
pub(super) fn chunked_backward<B: Backend>(
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
