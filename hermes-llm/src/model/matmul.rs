//! Mixed-precision matrix multiplication helpers.

use burn::prelude::*;
#[cfg(feature = "cuda")]
use burn::tensor::{DType, FloatDType};
use burn_nn::Linear;

#[cfg(feature = "cuda")]
fn matmul_dtype(device: &Device) -> Option<FloatDType> {
    // BF16 keeps FP32's exponent range during training; decode uses F16 for
    // compact prepared weights and equally fast A100 tensor-core matmuls.
    #[cfg(feature = "training-fusion")]
    let (dtype, float_dtype) = (DType::BF16, FloatDType::BF16);
    #[cfg(not(feature = "training-fusion"))]
    let (dtype, float_dtype) = (DType::F16, FloatDType::F16);

    device.supports_dtype(dtype).then_some(float_dtype)
}

pub(super) fn prepare_linear_for_inference(layer: &mut Linear) {
    #[cfg(feature = "cuda")]
    {
        let Some(dtype) = matmul_dtype(&layer.weight.val().device()) else {
            return;
        };
        layer.weight = layer.weight.clone().map(|weight| weight.cast(dtype));
        layer.bias = layer
            .bias
            .take()
            .map(|bias| bias.map(|value| value.cast(dtype)));
    }

    #[cfg(not(feature = "cuda"))]
    let _ = layer;
}

pub(super) fn matmul_input<const D: usize>(tensor: Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    {
        if let Some(dtype) = matmul_dtype(&tensor.device()) {
            return tensor.cast(dtype);
        }
    }

    tensor
}

#[cfg(feature = "cuda")]
fn linear_matmul<const D: usize>(layer: &Linear, input: Tensor<D>) -> Tensor<D> {
    burn::tensor::module::linear(
        matmul_input(input),
        matmul_input(layer.weight.val()),
        layer.bias.as_ref().map(|bias| matmul_input(bias.val())),
    )
}

pub(super) fn linear<const D: usize>(layer: &Linear, input: Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    {
        if matmul_dtype(&input.device()).is_some() {
            return linear_matmul(layer, input).cast(FloatDType::F32);
        }
    }

    layer.forward(input)
}

/// Run a training linear layer without promoting its output back to FP32.
///
/// This is useful between consecutive tensor-core projections when no FP32
/// residual operation occurs in between. Other backends and CUDA inference
/// retain the regular [`linear`] behavior.
pub(super) fn linear_low_precision<const D: usize>(layer: &Linear, input: Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "training-fusion")]
    {
        if matmul_dtype(&input.device()).is_some() {
            return linear_matmul(layer, input);
        }
    }

    linear(layer, input)
}

pub(super) fn matmul_2(lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
    #[cfg(feature = "cuda")]
    {
        if matmul_dtype(&lhs.device()).is_some() {
            return matmul_input(lhs)
                .matmul(matmul_input(rhs))
                .cast(FloatDType::F32);
        }
    }

    lhs.matmul(rhs)
}

#[cfg(feature = "training-fusion")]
pub(super) fn matmul_4(lhs: Tensor<4>, rhs: Tensor<4>) -> Tensor<4> {
    if matmul_dtype(&lhs.device()).is_some() {
        return matmul_input(lhs)
            .matmul(matmul_input(rhs))
            .cast(FloatDType::F32);
    }

    lhs.matmul(rhs)
}
