//! Fused linear head + cross-entropy loss.
//!
//! Layered like the scan family:
//!
//! - this module: the [`LinearCrossEntropyBackend`] extension trait and the
//!   shared vocabulary-padding contract;
//! - [`reference`]: the portable chunked tensor-op implementation — the
//!   correctness oracle and every non-GPU backend's path;
//! - [`autodiff`]: the first-order autodiff node;
//! - [`gpu`]: the CubeCL row-wise kernels (online log-sum-exp statistics
//!   and the fused gradient emission) and their dispatch;
//! - [`tests`]: CPU-vs-GPU parity, including padded-vocabulary gradients.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
use burn::backend::{
    Backend, Dispatch, NdArray, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::Int;
use burn_autodiff::Autodiff;

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

mod autodiff;
mod reference;

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu;

#[cfg(test)]
mod tests;
