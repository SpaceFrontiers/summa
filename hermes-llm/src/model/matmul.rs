//! Mixed-precision matrix multiplication helpers.

use burn::prelude::*;
#[cfg(feature = "cuda")]
use burn::tensor::{DType, FloatDType};
use burn_nn::Linear;

pub(super) fn matmul_input<const D: usize>(tensor: Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    {
        if tensor.device().supports_dtype(DType::F16) {
            return tensor.cast(FloatDType::F16);
        }
    }

    tensor
}

pub(super) fn linear<const D: usize>(layer: &Linear, input: Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    {
        if input.device().supports_dtype(DType::F16) {
            return burn::tensor::module::linear(
                matmul_input(input),
                matmul_input(layer.weight.val()),
                layer.bias.as_ref().map(|bias| matmul_input(bias.val())),
            )
            .cast(FloatDType::Flex32);
        }
    }

    layer.forward(input)
}

pub(super) fn matmul_2(lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
    #[cfg(feature = "cuda")]
    {
        if lhs.device().supports_dtype(DType::F16) {
            return matmul_input(lhs)
                .matmul(matmul_input(rhs))
                .cast(FloatDType::Flex32);
        }
    }

    lhs.matmul(rhs)
}

#[cfg(feature = "cuda")]
pub(super) fn matmul_4(lhs: Tensor<4>, rhs: Tensor<4>) -> Tensor<4> {
    if lhs.device().supports_dtype(DType::F16) {
        return matmul_input(lhs)
            .matmul(matmul_input(rhs))
            .cast(FloatDType::Flex32);
    }

    lhs.matmul(rhs)
}
