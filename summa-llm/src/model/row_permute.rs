//! Autodiff-aware row permutation for compact MoE dispatch.
//!
//! Burn's general `select` backward must scatter-add because indices may
//! repeat. MoE route ordering is a true permutation, so its inverse is both
//! exact and much cheaper: the backward is another row selection.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use burn::backend::NdArray;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::Int;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

#[cfg_attr(feature = "cuda", backend_extension(Cuda, Autodiff))]
#[cfg_attr(
    all(not(feature = "cuda"), feature = "metal"),
    backend_extension(Metal, Autodiff)
)]
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal")),
    backend_extension(NdArray, Autodiff)
)]
pub trait RowPermutationBackend: Backend {
    fn row_permute_inner(
        input: FloatTensor<Self>,
        order: IntTensor<Self>,
        _inverse: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        Self::float_select(input, 0, order)
    }
}

pub(super) fn row_permute(
    input: Tensor<2>,
    order: Tensor<1, Int>,
    inverse: Tensor<1, Int>,
) -> Tensor<2> {
    let [rows, _] = input.dims();
    assert_eq!(order.dims(), [rows]);
    assert_eq!(inverse.dims(), [rows]);
    if input.device() == Device::ndarray() {
        return input.select(0, order);
    }
    Tensor::from_dispatch(Dispatch::row_permute_inner(
        input.into_dispatch(),
        order.into_dispatch(),
        inverse.into_dispatch(),
    ))
}

impl RowPermutationBackend for burn_ndarray::NdArray {}

#[cfg(any(feature = "metal", feature = "cuda"))]
impl<R: burn_cubecl::CubeRuntime> RowPermutationBackend for burn_cubecl::CubeBackend<R> {}

#[cfg(feature = "training-fusion")]
impl<B: burn_fusion::FusionBackend> RowPermutationBackend for burn_fusion::Fusion<B> {}

#[derive(Clone, Debug)]
struct RowPermutationState<B: RowPermutationBackend> {
    order: IntTensor<B>,
    inverse: IntTensor<B>,
}

#[derive(Debug)]
struct RowPermutationBackward;

impl<B: RowPermutationBackend> Backward<B, 1> for RowPermutationBackward {
    type State = RowPermutationState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        if let Some(node) = node_input {
            let state = ops.state;
            let grad = B::row_permute_inner(grad, state.inverse, state.order);
            grads.register::<B>(node.id, grad);
        }
    }
}

impl<B: RowPermutationBackend, C: CheckpointStrategy> RowPermutationBackend for Autodiff<B, C> {
    fn row_permute_inner(
        input: FloatTensor<Self>,
        order: IntTensor<Self>,
        inverse: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let order = <Self as AutodiffBackend>::int_inner(order);
        let inverse = <Self as AutodiffBackend>::int_inner(inverse);
        match RowPermutationBackward
            .prepare::<C>([input.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::row_permute_inner(input.primitive, order.clone(), inverse.clone());
                prep.finish(RowPermutationState { order, inverse }, output)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::row_permute_inner(input.primitive, order, inverse))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    fn permutation_gradient(device: &Device) -> TensorData {
        let device = device.clone().autodiff();
        let input = Tensor::<2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
            &device,
        )
        .require_grad();
        let order = Tensor::<1, Int>::from_data(TensorData::new(vec![2_i64, 0, 1], [3]), &device);
        let inverse = Tensor::<1, Int>::from_data(TensorData::new(vec![1_i64, 2, 0], [3]), &device);
        let factors = Tensor::<2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
            &device,
        );
        let output = row_permute(input.clone(), order, inverse);
        let mut gradients = (output * factors).sum().backward();
        input
            .grad_remove(&mut gradients)
            .unwrap()
            .into_data()
            .convert::<f32>()
    }

    fn expected_gradient() -> TensorData {
        TensorData::new(vec![3.0_f32, 4.0, 5.0, 6.0, 1.0, 2.0], [3, 2])
    }

    #[test]
    fn permutation_backward_uses_the_inverse_order() {
        assert_eq!(
            permutation_gradient(&Device::ndarray()),
            expected_gradient()
        );
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    #[test]
    fn gpu_permutation_backward_uses_the_inverse_order() {
        assert_eq!(
            permutation_gradient(&crate::model::default_device()),
            expected_gradient()
        );
    }
}
