//! Fused Mamba selective scan.
//!
//! Layered like the other kernel families in this crate:
//!
//! - this module: the public entry point ([`selective_scan`]), the
//!   [`MambaBackend`] extension trait every backend implements, and the
//!   checkpointing constant the whole family agrees on;
//! - [`reference`]: a differentiable tensor-op implementation — the
//!   correctness oracle every GPU change is tested against, and the CPU
//!   backend's implementation;
//! - [`autodiff`]: the first-order autodiff node that saves the forward's
//!   checkpoints and routes gradients through the backend's fused backward;
//! - [`gpu`]: the CubeCL kernels (one swept forward, one swept backward,
//!   plus inference-path kernels) and their dispatch;
//! - [`tests`]: CPU-vs-GPU parity at production and deliberately awkward
//!   geometries.
//!
//! Size-generality and the measured state-width contract are documented in
//! `docs/kernel-tuning-surface.md`.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use burn::backend::NdArray;
use burn::backend::{Backend, Dispatch, backend_extension, tensor::FloatTensor};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn_autodiff::Autodiff;

use super::conv::DepthwiseConv1dBackend;
use reference::reference_selective_scan_tensor;

/// Recurrent-state checkpoint spacing for the GPU training scan: the
/// backward re-derives each segment's states from its entering checkpoint,
/// trading one recompute of the recurrence for a 32x smaller saved-state
/// tensor. Measured optimal at production scale (16 cost -2.3%, 8 -8.3%);
/// every sequence length works via partial-tail guards. The fusion IR and
/// both dispatch functions must agree on this value, which is why it is
/// the single source of the saved-states shape.
#[cfg(any(feature = "metal", feature = "cuda", test))]
pub(super) const CHECKPOINTED_SCAN_INTERVAL: usize = 32;

fn stable_softplus<const D: usize>(value: Tensor<D>) -> Tensor<D> {
    relu(value.clone()) + (-value.abs()).exp().add_scalar(1.0).log()
}

/// Backend capability required by the Mamba mixer.
#[cfg_attr(feature = "cuda", backend_extension(Cuda, Autodiff))]
#[cfg_attr(
    all(not(feature = "cuda"), feature = "metal"),
    backend_extension(Metal, Autodiff)
)]
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal")),
    backend_extension(NdArray, Autodiff)
)]
pub trait MambaBackend: Backend + DepthwiseConv1dBackend {
    #[allow(clippy::too_many_arguments)]
    fn selective_scan_inner(
        delta: FloatTensor<Self>,
        xs: FloatTensor<Self>,
        b_mat: FloatTensor<Self>,
        c_mat: FloatTensor<Self>,
        a: FloatTensor<Self>,
        d: FloatTensor<Self>,
        h: FloatTensor<Self>,
        state_dim: usize,
        save_states: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[allow(clippy::too_many_arguments, clippy::type_complexity, unused_variables)]
    fn selective_scan_backward(
        delta: FloatTensor<Self>,
        xs: FloatTensor<Self>,
        b_mat: FloatTensor<Self>,
        c_mat: FloatTensor<Self>,
        a: FloatTensor<Self>,
        d: FloatTensor<Self>,
        h: FloatTensor<Self>,
        states: FloatTensor<Self>,
        grad_y: FloatTensor<Self>,
        state_dim: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        panic!("selective scan only supports first-order autodiff")
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn selective_scan(
    delta: Tensor<3>,
    xs: Tensor<3>,
    b_mat: Tensor<3>,
    c_mat: Tensor<3>,
    a: Tensor<2>,
    d: Tensor<1>,
    h: Tensor<3>,
    state_dim: usize,
) -> (Tensor<3>, Tensor<3>) {
    if delta.device() == Device::ndarray() {
        let (y, h, _) =
            reference_selective_scan_tensor(delta, xs, b_mat, c_mat, a, d, h, state_dim, false);
        return (y, h);
    }
    let output = Dispatch::selective_scan_inner(
        delta.into_dispatch(),
        xs.into_dispatch(),
        b_mat.into_dispatch(),
        c_mat.into_dispatch(),
        a.into_dispatch(),
        d.into_dispatch(),
        h.into_dispatch(),
        state_dim,
        false,
    );
    (
        Tensor::from_dispatch(output.0),
        Tensor::from_dispatch(output.1),
    )
}

mod autodiff;
mod reference;

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu;

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod tests;
