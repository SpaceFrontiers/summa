//! Fused SwiGLU activation for compact MoE expert rows.
//!
//! Burn's generic SiLU expression promotes BF16 to FP32 and expands into
//! several elementwise kernels. The MoE projection has a fixed
//! `[gate | value]` layout, so one CUDA kernel can evaluate the activation and
//! product while retaining FP32 intermediate math. The custom autodiff node
//! likewise computes both input-gradient halves in one launch.

use burn::backend::Cuda;
use burn::backend::{Backend, Dispatch, backend_extension, tensor::FloatTensor};
use burn::prelude::*;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

#[backend_extension(Cuda, Autodiff)]
pub trait FusedSwiGluBackend: Backend {
    fn fused_swiglu_inner(input: FloatTensor<Self>, intermediate: usize) -> FloatTensor<Self>;

    fn fused_swiglu_backward(
        _input: FloatTensor<Self>,
        _grad: FloatTensor<Self>,
        _intermediate: usize,
    ) -> FloatTensor<Self> {
        panic!("fused SwiGLU only supports first-order autodiff")
    }
}

pub(super) fn fused_swiglu(input: Tensor<2>, intermediate: usize) -> Tensor<2> {
    let [rows, columns] = input.dims();
    assert!(rows > 0, "fused SwiGLU requires at least one row");
    assert_eq!(
        columns,
        intermediate * 2,
        "fused SwiGLU expects a [gate | value] projection"
    );
    Tensor::from_dispatch(Dispatch::fused_swiglu_inner(
        input.into_dispatch(),
        intermediate,
    ))
}

#[derive(Clone, Debug)]
struct FusedSwiGluState<B: FusedSwiGluBackend> {
    input: FloatTensor<B>,
    intermediate: usize,
}

#[derive(Debug)]
struct FusedSwiGluBackward;

impl<B: FusedSwiGluBackend> Backward<B, 1> for FusedSwiGluBackward {
    type State = FusedSwiGluState<B>;

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
            let grad = B::fused_swiglu_backward(state.input, grad, state.intermediate);
            grads.register::<B>(node.id, grad);
        }
    }
}

impl<B: FusedSwiGluBackend, C: CheckpointStrategy> FusedSwiGluBackend for Autodiff<B, C> {
    fn fused_swiglu_inner(input: FloatTensor<Self>, intermediate: usize) -> FloatTensor<Self> {
        match FusedSwiGluBackward
            .prepare::<C>([input.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::fused_swiglu_inner(input.primitive.clone(), intermediate);
                prep.finish(
                    FusedSwiGluState {
                        input: input.primitive,
                        intermediate,
                    },
                    output,
                )
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::fused_swiglu_inner(input.primitive, intermediate))
            }
        }
    }
}

mod gpu {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};
    use half::{bf16, f16};

    use super::FusedSwiGluBackend;
    use crate::model::cube_tensor::{empty_like, into_contiguous};

    const THREADS: u32 = 256;

    #[cube(launch)]
    fn swiglu_forward<F: Float>(
        input: &Tensor<F>,
        output: &mut Tensor<F>,
        elements: u32,
        intermediate: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < elements as usize {
            let intermediate = intermediate as usize;
            let row = index / intermediate;
            let column = index % intermediate;
            let gate = f32::cast_from(input[row * intermediate * 2 + column]);
            let value = f32::cast_from(input[row * intermediate * 2 + intermediate + column]);
            let sigmoid = 1.0f32 / (1.0f32 + (-gate).exp());
            output[index] = F::cast_from(gate * sigmoid * value);
        }
    }

    #[cube(launch)]
    fn swiglu_backward<F: Float>(
        input: &Tensor<F>,
        grad: &Tensor<F>,
        grad_input: &mut Tensor<F>,
        elements: u32,
        intermediate: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < elements as usize {
            let intermediate = intermediate as usize;
            let row = index / intermediate;
            let column = index % intermediate;
            let gate_index = row * intermediate * 2 + column;
            let value_index = gate_index + intermediate;
            let gate = f32::cast_from(input[gate_index]);
            let value = f32::cast_from(input[value_index]);
            let grad = f32::cast_from(grad[index]);
            let sigmoid = 1.0f32 / (1.0f32 + (-gate).exp());
            let silu = gate * sigmoid;
            let silu_gradient = sigmoid * (1.0f32 + gate * (1.0f32 - sigmoid));
            grad_input[gate_index] = F::cast_from(grad * value * silu_gradient);
            grad_input[value_index] = F::cast_from(grad * silu);
        }
    }

    fn launch_forward<F: Float + CubeElement, R: CubeRuntime>(
        input: CubeTensor<R>,
        intermediate: usize,
    ) -> CubeTensor<R> {
        let input = into_contiguous(input);
        let [rows, columns] = input.meta.shape.dims();
        assert_eq!(columns, intermediate * 2);
        let output = empty_like(&input, Shape::new([rows, intermediate]));
        let elements = rows
            .checked_mul(intermediate)
            .expect("fused SwiGLU element count overflow");
        let client = input.client.clone();
        swiglu_forward::launch::<F, R>(
            &client,
            CubeCount::Static((elements as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            elements as u32,
            intermediate as u32,
        );
        output
    }

    fn launch_backward<F: Float + CubeElement, R: CubeRuntime>(
        input: CubeTensor<R>,
        grad: CubeTensor<R>,
        intermediate: usize,
    ) -> CubeTensor<R> {
        let input = into_contiguous(input);
        let grad = into_contiguous(grad);
        let [rows, columns] = input.meta.shape.dims();
        assert_eq!(columns, intermediate * 2);
        assert_eq!(grad.meta.shape.dims::<2>(), [rows, intermediate]);
        assert_eq!(input.dtype, grad.dtype);
        let grad_input = empty_like(&input, input.meta.shape.clone());
        let elements = rows
            .checked_mul(intermediate)
            .expect("fused SwiGLU element count overflow");
        let client = input.client.clone();
        swiglu_backward::launch::<F, R>(
            &client,
            CubeCount::Static((elements as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            input.into_tensor_arg(),
            grad.into_tensor_arg(),
            grad_input.clone().into_tensor_arg(),
            elements as u32,
            intermediate as u32,
        );
        grad_input
    }

    impl<R: CubeRuntime> FusedSwiGluBackend for CubeBackend<R> {
        fn fused_swiglu_inner(input: CubeTensor<R>, intermediate: usize) -> CubeTensor<R> {
            match input.dtype {
                DType::BF16 => launch_forward::<bf16, R>(input, intermediate),
                DType::F16 => launch_forward::<f16, R>(input, intermediate),
                DType::F32 => launch_forward::<f32, R>(input, intermediate),
                dtype => panic!("fused SwiGLU needs a floating dtype, got {dtype:?}"),
            }
        }

        fn fused_swiglu_backward(
            input: CubeTensor<R>,
            grad: CubeTensor<R>,
            intermediate: usize,
        ) -> CubeTensor<R> {
            match input.dtype {
                DType::BF16 => launch_backward::<bf16, R>(input, grad, intermediate),
                DType::F16 => launch_backward::<f16, R>(input, grad, intermediate),
                DType::F32 => launch_backward::<f32, R>(input, grad, intermediate),
                dtype => panic!("fused SwiGLU needs a floating dtype, got {dtype:?}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{FloatDType, TensorData, activation::silu};

    use super::*;

    fn run(fused: bool) -> (Vec<f32>, Vec<f32>) {
        let device = crate::model::default_device().autodiff();
        let input = Tensor::<2>::from_data(
            TensorData::new(
                (0..48)
                    .map(|index| (index as f32 * 0.17).sin() * 2.0)
                    .collect::<Vec<_>>(),
                [4, 12],
            ),
            &device,
        )
        .cast(FloatDType::BF16)
        .require_grad();
        let output = if fused {
            fused_swiglu(input.clone(), 6)
        } else {
            let gate = input.clone().slice([0..4, 0..6]);
            let value = input.clone().slice([0..4, 6..12]);
            silu(gate) * value
        };
        let factors = Tensor::<2>::from_data(
            TensorData::new(
                (0..24)
                    .map(|index| 0.3 + (index as f32 * 0.11).cos())
                    .collect::<Vec<_>>(),
                [4, 6],
            ),
            &device,
        )
        .cast(FloatDType::BF16);
        let values = output
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let mut gradients = (output * factors).sum().backward();
        let gradient = input
            .grad_remove(&mut gradients)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        (values, gradient)
    }

    #[test]
    fn fused_swiglu_matches_tensor_expression() {
        let actual = run(true);
        let expected = run(false);
        for (name, actual, expected) in [
            ("output", actual.0, expected.0),
            ("gradient", actual.1, expected.1),
        ] {
            let difference = actual
                .into_iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f32::max);
            assert!(difference <= 0.03, "{name} max difference {difference}");
        }
    }
}
