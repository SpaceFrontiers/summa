//! Fused compact MoE dispatch and route combination.
//!
//! Generic tensor expressions first repeat every token `top_k` times, permute
//! those rows, then permute them back before a weighted reduction. These CUDA
//! kernels address the compact routed rows directly. Besides reducing memory
//! traffic, their custom autodiff nodes use the known one-to-one permutation
//! instead of general scatter operations.

use burn::backend::Cuda;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch, TensorMetadata, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::Int;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

#[backend_extension(Cuda, Autodiff)]
pub trait MoeDispatchBackend: Backend {
    fn moe_route_gather(
        input: FloatTensor<Self>,
        order: IntTensor<Self>,
        inverse: IntTensor<Self>,
        top_k: usize,
    ) -> FloatTensor<Self>;

    fn moe_route_gather_backward(
        _grad: FloatTensor<Self>,
        _inverse: IntTensor<Self>,
        _tokens: usize,
        _top_k: usize,
    ) -> FloatTensor<Self> {
        panic!("MoE route gather only supports first-order autodiff")
    }

    fn moe_route_combine(
        routed: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        inverse: IntTensor<Self>,
        top_k: usize,
    ) -> FloatTensor<Self>;

    fn moe_route_combine_backward(
        _routed: FloatTensor<Self>,
        _weights: FloatTensor<Self>,
        _inverse: IntTensor<Self>,
        _grad: FloatTensor<Self>,
        _top_k: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        panic!("MoE route combine only supports first-order autodiff")
    }
}

pub(super) fn route_gather(
    input: Tensor<2>,
    order: Tensor<1, Int>,
    inverse: Tensor<1, Int>,
    top_k: usize,
) -> Tensor<2> {
    let [tokens, _] = input.dims();
    let routes = tokens.checked_mul(top_k).expect("MoE route count overflow");
    assert_eq!(order.dims(), [routes]);
    assert_eq!(inverse.dims(), [routes]);
    Tensor::from_dispatch(Dispatch::moe_route_gather(
        input.into_dispatch(),
        order.into_dispatch(),
        inverse.into_dispatch(),
        top_k,
    ))
}

pub(super) fn route_combine(
    routed: Tensor<2>,
    weights: Tensor<2>,
    inverse: Tensor<1, Int>,
    top_k: usize,
) -> Tensor<2> {
    let [routes, _] = routed.dims();
    let [tokens, weight_top_k] = weights.dims();
    assert_eq!(weight_top_k, top_k);
    assert_eq!(routes, tokens * top_k);
    assert_eq!(inverse.dims(), [routes]);
    Tensor::from_dispatch(Dispatch::moe_route_combine(
        routed.into_dispatch(),
        weights.into_dispatch(),
        inverse.into_dispatch(),
        top_k,
    ))
}

#[derive(Clone, Debug)]
struct RouteGatherState<B: MoeDispatchBackend> {
    inverse: IntTensor<B>,
    tokens: usize,
    top_k: usize,
}

#[derive(Debug)]
struct RouteGatherBackward;

impl<B: MoeDispatchBackend> Backward<B, 1> for RouteGatherBackward {
    type State = RouteGatherState<B>;

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
            let grad = B::moe_route_gather_backward(grad, state.inverse, state.tokens, state.top_k);
            grads.register::<B>(node.id, grad);
        }
    }
}

#[derive(Clone, Debug)]
struct RouteCombineState<B: MoeDispatchBackend> {
    routed: FloatTensor<B>,
    weights: FloatTensor<B>,
    inverse: IntTensor<B>,
    top_k: usize,
}

#[derive(Debug)]
struct RouteCombineBackward;

impl<B: MoeDispatchBackend> Backward<B, 2> for RouteCombineBackward {
    type State = RouteCombineState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_routed, node_weights] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let (grad_routed, grad_weights) = B::moe_route_combine_backward(
            state.routed,
            state.weights,
            state.inverse,
            grad,
            state.top_k,
        );
        if let Some(node) = node_routed {
            grads.register::<B>(node.id, grad_routed);
        }
        if let Some(node) = node_weights {
            grads.register::<B>(node.id, grad_weights);
        }
    }
}

impl<B: MoeDispatchBackend, C: CheckpointStrategy> MoeDispatchBackend for Autodiff<B, C> {
    fn moe_route_gather(
        input: FloatTensor<Self>,
        order: IntTensor<Self>,
        inverse: IntTensor<Self>,
        top_k: usize,
    ) -> FloatTensor<Self> {
        let order = <Self as AutodiffBackend>::int_inner(order);
        let inverse = <Self as AutodiffBackend>::int_inner(inverse);
        let tokens = input.primitive.shape().dims::<2>()[0];
        match RouteGatherBackward
            .prepare::<C>([input.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::moe_route_gather(input.primitive, order, inverse.clone(), top_k);
                prep.finish(
                    RouteGatherState {
                        inverse,
                        tokens,
                        top_k,
                    },
                    output,
                )
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::moe_route_gather(input.primitive, order, inverse, top_k))
            }
        }
    }

    fn moe_route_combine(
        routed: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        inverse: IntTensor<Self>,
        top_k: usize,
    ) -> FloatTensor<Self> {
        let inverse = <Self as AutodiffBackend>::int_inner(inverse);
        match RouteCombineBackward
            .prepare::<C>([routed.node.clone(), weights.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::moe_route_combine(
                    routed.primitive.clone(),
                    weights.primitive.clone(),
                    inverse.clone(),
                    top_k,
                );
                prep.finish(
                    RouteCombineState {
                        routed: routed.primitive,
                        weights: weights.primitive,
                        inverse,
                        top_k,
                    },
                    output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::moe_route_combine(
                routed.primitive,
                weights.primitive,
                inverse,
                top_k,
            )),
        }
    }
}

mod gpu {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};
    use half::{bf16, f16};

    use super::MoeDispatchBackend;
    use crate::model::cube_tensor::{empty_like, empty_like_dtype, into_contiguous};

    const THREADS: u32 = 256;

    #[cube(launch)]
    fn route_gather_forward<F: Float>(
        input: &Tensor<F>,
        order: &Tensor<i32>,
        output: &mut Tensor<F>,
        elements: u32,
        hidden: u32,
        top_k: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < elements as usize {
            let hidden = hidden as usize;
            let route = index / hidden;
            let column = index % hidden;
            let original_route = order[route] as usize;
            let token = original_route / top_k as usize;
            output[index] = input[token * hidden + column];
        }
    }

    #[cube(launch)]
    fn route_gather_backward<F: Float>(
        grad: &Tensor<F>,
        inverse: &Tensor<i32>,
        grad_input: &mut Tensor<F>,
        elements: u32,
        hidden: u32,
        top_k: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < elements as usize {
            let hidden = hidden as usize;
            let token = index / hidden;
            let column = index % hidden;
            let mut sum = f32::cast_from(0.0f32);
            for choice in 0..top_k as usize {
                let sorted = inverse[token * top_k as usize + choice] as usize;
                sum += f32::cast_from(grad[sorted * hidden + column]);
            }
            grad_input[index] = F::cast_from(sum);
        }
    }

    #[cube(launch)]
    fn route_combine_forward<F: Float>(
        routed: &Tensor<F>,
        weights: &Tensor<f32>,
        inverse: &Tensor<i32>,
        output: &mut Tensor<F>,
        elements: u32,
        hidden: u32,
        top_k: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < elements as usize {
            let hidden = hidden as usize;
            let token = index / hidden;
            let column = index % hidden;
            let mut sum = f32::cast_from(0.0f32);
            for choice in 0..top_k as usize {
                let route = token * top_k as usize + choice;
                let sorted = inverse[route] as usize;
                sum += f32::cast_from(routed[sorted * hidden + column]) * weights[route];
            }
            output[index] = F::cast_from(sum);
        }
    }

    #[cube(launch)]
    fn route_combine_routed_backward<F: Float>(
        weights: &Tensor<f32>,
        inverse: &Tensor<i32>,
        grad: &Tensor<F>,
        grad_routed: &mut Tensor<F>,
        elements: u32,
        hidden: u32,
        top_k: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < elements as usize {
            let hidden = hidden as usize;
            let original_route = index / hidden;
            let column = index % hidden;
            let sorted = inverse[original_route] as usize;
            let token = original_route / top_k as usize;
            let value = f32::cast_from(grad[token * hidden + column]) * weights[original_route];
            grad_routed[sorted * hidden + column] = F::cast_from(value);
        }
    }

    #[cube(launch)]
    fn route_combine_weight_backward<F: Float>(
        routed: &Tensor<F>,
        inverse: &Tensor<i32>,
        grad: &Tensor<F>,
        grad_weights: &mut Tensor<f32>,
        hidden: u32,
        top_k: u32,
    ) {
        let route = CUBE_POS_X as usize;
        let lane = UNIT_POS_X as usize;
        let hidden = hidden as usize;
        let sorted = inverse[route] as usize;
        let token = route / top_k as usize;
        let mut partial = f32::cast_from(0.0f32);
        let mut column = lane;
        while column < hidden {
            partial += f32::cast_from(grad[token * hidden + column])
                * f32::cast_from(routed[sorted * hidden + column]);
            column += THREADS as usize;
        }
        let mut reduction = Shared::new_slice(THREADS as usize);
        reduction[lane] = partial;
        sync_cube();
        let mut distance = 128u32;
        while distance > 0 {
            let current = reduction[lane];
            let neighbor = if lane < distance as usize {
                reduction[lane + distance as usize]
            } else {
                0.0f32
            };
            sync_cube();
            if lane < distance as usize {
                reduction[lane] = current + neighbor;
            }
            sync_cube();
            distance /= 2;
        }
        if lane == 0 {
            grad_weights[route] = reduction[0];
        }
    }

    fn launch_gather<F: Float + CubeElement, R: CubeRuntime>(
        input: CubeTensor<R>,
        order: CubeTensor<R>,
        top_k: usize,
    ) -> CubeTensor<R> {
        let input = into_contiguous(input);
        let order = into_contiguous(order);
        let [tokens, hidden] = input.meta.shape.dims();
        let routes = tokens * top_k;
        assert_eq!(order.meta.shape.dims::<1>(), [routes]);
        let output = empty_like(&input, Shape::new([routes, hidden]));
        let elements = routes * hidden;
        let client = input.client.clone();
        route_gather_forward::launch::<F, R>(
            &client,
            CubeCount::Static((elements as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            input.into_tensor_arg(),
            order.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            elements as u32,
            hidden as u32,
            top_k as u32,
        );
        output
    }

    fn launch_gather_backward<F: Float + CubeElement, R: CubeRuntime>(
        grad: CubeTensor<R>,
        inverse: CubeTensor<R>,
        tokens: usize,
        top_k: usize,
    ) -> CubeTensor<R> {
        let grad = into_contiguous(grad);
        let inverse = into_contiguous(inverse);
        let [routes, hidden] = grad.meta.shape.dims();
        assert_eq!(routes, tokens * top_k);
        let grad_input = empty_like(&grad, Shape::new([tokens, hidden]));
        let elements = tokens * hidden;
        let client = grad.client.clone();
        route_gather_backward::launch::<F, R>(
            &client,
            CubeCount::Static((elements as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            grad.into_tensor_arg(),
            inverse.into_tensor_arg(),
            grad_input.clone().into_tensor_arg(),
            elements as u32,
            hidden as u32,
            top_k as u32,
        );
        grad_input
    }

    fn launch_combine<F: Float + CubeElement, R: CubeRuntime>(
        routed: CubeTensor<R>,
        weights: CubeTensor<R>,
        inverse: CubeTensor<R>,
        top_k: usize,
    ) -> CubeTensor<R> {
        let routed = into_contiguous(routed);
        let weights = into_contiguous(weights);
        let inverse = into_contiguous(inverse);
        let [routes, hidden] = routed.meta.shape.dims();
        let [tokens, weight_top_k] = weights.meta.shape.dims();
        assert_eq!(weight_top_k, top_k);
        assert_eq!(routes, tokens * top_k);
        assert_eq!(weights.dtype, DType::F32, "MoE route weights must be FP32");
        let output = empty_like(&routed, Shape::new([tokens, hidden]));
        let elements = tokens * hidden;
        let client = routed.client.clone();
        route_combine_forward::launch::<F, R>(
            &client,
            CubeCount::Static((elements as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            routed.into_tensor_arg(),
            weights.into_tensor_arg(),
            inverse.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            elements as u32,
            hidden as u32,
            top_k as u32,
        );
        output
    }

    fn launch_combine_backward<F: Float + CubeElement, R: CubeRuntime>(
        routed: CubeTensor<R>,
        weights: CubeTensor<R>,
        inverse: CubeTensor<R>,
        grad: CubeTensor<R>,
        top_k: usize,
    ) -> (CubeTensor<R>, CubeTensor<R>) {
        let routed = into_contiguous(routed);
        let weights = into_contiguous(weights);
        let inverse = into_contiguous(inverse);
        let grad = into_contiguous(grad);
        let [routes, hidden] = routed.meta.shape.dims();
        let [tokens, weight_top_k] = weights.meta.shape.dims();
        assert_eq!(weight_top_k, top_k);
        assert_eq!(routes, tokens * top_k);
        assert_eq!(grad.meta.shape.dims::<2>(), [tokens, hidden]);
        assert_eq!(weights.dtype, DType::F32, "MoE route weights must be FP32");
        let grad_routed = empty_like(&routed, routed.meta.shape.clone());
        let grad_weights = empty_like_dtype(&weights, weights.meta.shape.clone(), DType::F32);
        let routed_elements = routes * hidden;
        let client = routed.client.clone();
        route_combine_routed_backward::launch::<F, R>(
            &client,
            CubeCount::Static((routed_elements as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            weights.clone().into_tensor_arg(),
            inverse.clone().into_tensor_arg(),
            grad.clone().into_tensor_arg(),
            grad_routed.clone().into_tensor_arg(),
            routed_elements as u32,
            hidden as u32,
            top_k as u32,
        );
        route_combine_weight_backward::launch::<F, R>(
            &client,
            CubeCount::Static(routes as u32, 1, 1),
            CubeDim::new_1d(THREADS),
            routed.into_tensor_arg(),
            inverse.into_tensor_arg(),
            grad.into_tensor_arg(),
            grad_weights.clone().into_tensor_arg(),
            hidden as u32,
            top_k as u32,
        );
        (grad_routed, grad_weights)
    }

    impl<R: CubeRuntime> MoeDispatchBackend for CubeBackend<R> {
        fn moe_route_gather(
            input: CubeTensor<R>,
            order: CubeTensor<R>,
            _inverse: CubeTensor<R>,
            top_k: usize,
        ) -> CubeTensor<R> {
            match input.dtype {
                DType::BF16 => launch_gather::<bf16, R>(input, order, top_k),
                DType::F16 => launch_gather::<f16, R>(input, order, top_k),
                DType::F32 => launch_gather::<f32, R>(input, order, top_k),
                dtype => panic!("MoE dispatch needs a floating dtype, got {dtype:?}"),
            }
        }

        fn moe_route_gather_backward(
            grad: CubeTensor<R>,
            inverse: CubeTensor<R>,
            tokens: usize,
            top_k: usize,
        ) -> CubeTensor<R> {
            match grad.dtype {
                DType::BF16 => launch_gather_backward::<bf16, R>(grad, inverse, tokens, top_k),
                DType::F16 => launch_gather_backward::<f16, R>(grad, inverse, tokens, top_k),
                DType::F32 => launch_gather_backward::<f32, R>(grad, inverse, tokens, top_k),
                dtype => panic!("MoE dispatch needs a floating dtype, got {dtype:?}"),
            }
        }

        fn moe_route_combine(
            routed: CubeTensor<R>,
            weights: CubeTensor<R>,
            inverse: CubeTensor<R>,
            top_k: usize,
        ) -> CubeTensor<R> {
            match routed.dtype {
                DType::BF16 => launch_combine::<bf16, R>(routed, weights, inverse, top_k),
                DType::F16 => launch_combine::<f16, R>(routed, weights, inverse, top_k),
                DType::F32 => launch_combine::<f32, R>(routed, weights, inverse, top_k),
                dtype => panic!("MoE dispatch needs a floating dtype, got {dtype:?}"),
            }
        }

        fn moe_route_combine_backward(
            routed: CubeTensor<R>,
            weights: CubeTensor<R>,
            inverse: CubeTensor<R>,
            grad: CubeTensor<R>,
            top_k: usize,
        ) -> (CubeTensor<R>, CubeTensor<R>) {
            match routed.dtype {
                DType::BF16 => {
                    launch_combine_backward::<bf16, R>(routed, weights, inverse, grad, top_k)
                }
                DType::F16 => {
                    launch_combine_backward::<f16, R>(routed, weights, inverse, grad, top_k)
                }
                DType::F32 => {
                    launch_combine_backward::<f32, R>(routed, weights, inverse, grad, top_k)
                }
                dtype => panic!("MoE dispatch needs a floating dtype, got {dtype:?}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{FloatDType, TensorData};

    use super::*;

    fn run(fused: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let device = crate::model::default_device().autodiff();
        let input = Tensor::<2>::from_data(
            TensorData::new(
                (0..20)
                    .map(|index| (index as f32 * 0.19).sin())
                    .collect::<Vec<_>>(),
                [5, 4],
            ),
            &device,
        )
        .cast(FloatDType::BF16)
        .require_grad();
        let order = Tensor::<1, Int>::from_data(
            TensorData::new(vec![1_i64, 4, 6, 9, 2, 3, 5, 8, 0, 7], [10]),
            &device,
        );
        let mut inverse_values = vec![0_i64; 10];
        for (sorted, original) in [1_usize, 4, 6, 9, 2, 3, 5, 8, 0, 7].into_iter().enumerate() {
            inverse_values[original] = sorted as i64;
        }
        let inverse = Tensor::<1, Int>::from_data(TensorData::new(inverse_values, [10]), &device);
        let weights = Tensor::<2>::from_data(
            TensorData::new(
                vec![0.7_f32, 0.3, 0.2, 0.8, 0.6, 0.4, 0.1, 0.9, 0.55, 0.45],
                [5, 2],
            ),
            &device,
        )
        .require_grad();
        let output = if fused {
            let routed = route_gather(input.clone(), order, inverse.clone(), 2);
            route_combine(routed.square(), weights.clone(), inverse, 2)
        } else {
            let repeated = input
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, 2)
                .reshape([10, 4]);
            (repeated.square().reshape([5, 2, 4])
                * weights.clone().cast(FloatDType::BF16).reshape([5, 2, 1]))
            .sum_dim(1)
            .reshape([5, 4])
        };
        let values = output
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let mut gradients = output.sum().backward();
        let input_gradient = input
            .grad_remove(&mut gradients)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let weight_gradient = weights
            .grad_remove(&mut gradients)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        (values, input_gradient, weight_gradient)
    }

    #[test]
    fn fused_dispatch_and_combine_match_tensor_expression() {
        let actual = run(true);
        let expected = run(false);
        for (name, actual, expected) in [
            ("output", actual.0, expected.0),
            ("input gradient", actual.1, expected.1),
            ("weight gradient", actual.2, expected.2),
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
