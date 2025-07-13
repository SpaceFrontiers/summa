//! Mixed-precision matrix multiplication helpers.

use burn::prelude::*;
#[cfg(feature = "cuda")]
use burn::tensor::{DType, FloatDType};
use burn_nn::Linear;

pub(super) fn matmul_input<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    #[cfg(feature = "cuda")]
    {
        if B::supports_dtype(&tensor.device(), DType::F16) {
            return tensor.cast(FloatDType::F16);
        }
    }

    tensor
}

pub(super) fn linear<B: Backend, const D: usize>(
    layer: &Linear<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    #[cfg(feature = "cuda")]
    {
        if B::supports_dtype(&input.device(), DType::F16) {
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

pub(super) fn matmul_2<B: Backend>(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
    #[cfg(feature = "cuda")]
    {
        if B::supports_dtype(&lhs.device(), DType::F16) {
            return matmul_input(lhs)
                .matmul(matmul_input(rhs))
                .cast(FloatDType::Flex32);
        }
    }

    lhs.matmul(rhs)
}

#[cfg(feature = "cuda")]
pub(super) fn matmul_4<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Tensor<B, 4> {
    if B::supports_dtype(&lhs.device(), DType::F16) {
        return matmul_input(lhs)
            .matmul(matmul_input(rhs))
            .cast(FloatDType::Flex32);
    }

    lhs.matmul(rhs)
}
