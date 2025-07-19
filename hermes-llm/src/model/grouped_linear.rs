//! Differentiable compact grouped linear projections for MoE experts.
//!
//! CUDA 12.5+ on SM80+ dispatches all nonempty expert matrices through one
//! native grouped GEMM. Other CUDA devices retain an ordinary per-expert
//! tensor-op fallback, so enabling MoE does not raise the model's hardware
//! floor.

use burn::backend::Cuda;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch, DispatchDevice, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::Int;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use core::sync::atomic::{AtomicU64, Ordering};

static NEXT_GROUPED_MLP_CACHE_KEY: AtomicU64 = AtomicU64::new(1);

#[backend_extension(Cuda, Autodiff)]
pub trait GroupedLinearBackend: Backend {
    fn grouped_linear_inner(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        counts: Vec<usize>,
    ) -> FloatTensor<Self>;

    fn grouped_linear_backward(
        _input: FloatTensor<Self>,
        _weight: FloatTensor<Self>,
        _grad: FloatTensor<Self>,
        _counts: Vec<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        panic!("grouped linear only supports first-order autodiff")
    }

    fn grouped_swiglu_mlp_inner(
        input: FloatTensor<Self>,
        in_weight: FloatTensor<Self>,
        down_weight: FloatTensor<Self>,
        counts: IntTensor<Self>,
        intermediate: usize,
        cache_key: Option<u64>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[allow(clippy::too_many_arguments)]
    fn grouped_swiglu_mlp_backward(
        _input: FloatTensor<Self>,
        _in_weight: FloatTensor<Self>,
        _down_weight: FloatTensor<Self>,
        _counts: IntTensor<Self>,
        _projected: FloatTensor<Self>,
        _hidden: FloatTensor<Self>,
        _grad: FloatTensor<Self>,
        _intermediate: usize,
        _cache_key: Option<u64>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        panic!("grouped SwiGLU MLP only supports first-order autodiff")
    }
}

pub(super) fn grouped_linear(input: Tensor<2>, weight: Tensor<3>, counts: &[usize]) -> Tensor<2> {
    let [routes, input_width] = input.dims();
    let [experts, weight_input, _] = weight.dims();
    assert_eq!(
        counts.len(),
        experts,
        "one route count is required per expert"
    );
    assert_eq!(
        counts.iter().sum::<usize>(),
        routes,
        "route counts must cover the input"
    );
    assert_eq!(
        input_width, weight_input,
        "grouped linear input width mismatch"
    );
    Tensor::from_dispatch(Dispatch::grouped_linear_inner(
        input.into_dispatch(),
        weight.into_dispatch(),
        counts.to_vec(),
    ))
}

pub(super) fn grouped_swiglu_mlp(
    input: Tensor<2>,
    in_weight: Tensor<3>,
    down_weight: Tensor<3>,
    counts: Tensor<1, Int>,
    intermediate: usize,
) -> Tensor<2> {
    let [routes, input_width] = input.dims();
    let [experts, in_weight_input, projected_width] = in_weight.dims();
    let [down_experts, down_input, output_width] = down_weight.dims();
    assert_eq!(in_weight_input, input_width);
    assert_eq!(projected_width, intermediate * 2);
    assert_eq!(down_experts, experts);
    assert_eq!(down_input, intermediate);
    assert_eq!(counts.dims(), [experts]);
    let (output, _, _) = Dispatch::grouped_swiglu_mlp_inner(
        input.into_dispatch(),
        in_weight.into_dispatch(),
        down_weight.into_dispatch(),
        counts.into_dispatch(),
        intermediate,
        None,
    );
    let output = Tensor::from_dispatch(output);
    assert_eq!(output.dims(), [routes, output_width]);
    output
}

pub(super) fn is_cuda_device(device: &Device) -> bool {
    match device.as_dispatch() {
        DispatchDevice::Cuda(_) => true,
        DispatchDevice::Autodiff(device) => matches!(&**device, DispatchDevice::Cuda(_)),
        _ => false,
    }
}

#[derive(Clone, Debug)]
struct GroupedLinearState<B: GroupedLinearBackend> {
    input: FloatTensor<B>,
    weight: FloatTensor<B>,
    counts: Vec<usize>,
}

#[derive(Debug)]
struct GroupedLinearBackward;

impl<B: GroupedLinearBackend> Backward<B, 2> for GroupedLinearBackward {
    type State = GroupedLinearState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input, node_weight] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let (grad_input, grad_weight) =
            B::grouped_linear_backward(state.input, state.weight, grad, state.counts);
        if let Some(node) = node_input {
            grads.register::<B>(node.id, grad_input);
        }
        if let Some(node) = node_weight {
            grads.register::<B>(node.id, grad_weight);
        }
    }
}

impl<B: GroupedLinearBackend, C: CheckpointStrategy> GroupedLinearBackend for Autodiff<B, C> {
    fn grouped_linear_inner(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        counts: Vec<usize>,
    ) -> FloatTensor<Self> {
        match GroupedLinearBackward
            .prepare::<C>([input.node.clone(), weight.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::grouped_linear_inner(
                    input.primitive.clone(),
                    weight.primitive.clone(),
                    counts.clone(),
                );
                prep.finish(
                    GroupedLinearState {
                        input: input.primitive,
                        weight: weight.primitive,
                        counts,
                    },
                    output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::grouped_linear_inner(
                input.primitive,
                weight.primitive,
                counts,
            )),
        }
    }

    fn grouped_swiglu_mlp_inner(
        input: FloatTensor<Self>,
        in_weight: FloatTensor<Self>,
        down_weight: FloatTensor<Self>,
        counts: IntTensor<Self>,
        intermediate: usize,
        _cache_key: Option<u64>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let counts = <Self as AutodiffBackend>::int_inner(counts);
        match GroupedSwiGluMlpBackward
            .prepare::<C>([
                input.node.clone(),
                in_weight.node.clone(),
                down_weight.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let cache_key = NEXT_GROUPED_MLP_CACHE_KEY.fetch_add(1, Ordering::Relaxed);
                let (output, projected, hidden) = B::grouped_swiglu_mlp_inner(
                    input.primitive.clone(),
                    in_weight.primitive.clone(),
                    down_weight.primitive.clone(),
                    counts.clone(),
                    intermediate,
                    Some(cache_key),
                );
                let output = prep.finish(
                    GroupedSwiGluMlpState {
                        input: input.primitive,
                        in_weight: in_weight.primitive,
                        down_weight: down_weight.primitive,
                        counts,
                        projected: projected.clone(),
                        hidden: hidden.clone(),
                        intermediate,
                        cache_key: Some(cache_key),
                    },
                    output,
                );
                (
                    output,
                    <Self as AutodiffBackend>::from_inner(projected),
                    <Self as AutodiffBackend>::from_inner(hidden),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (output, projected, hidden) = B::grouped_swiglu_mlp_inner(
                    input.primitive,
                    in_weight.primitive,
                    down_weight.primitive,
                    counts,
                    intermediate,
                    None,
                );
                (
                    prep.finish(output),
                    <Self as AutodiffBackend>::from_inner(projected),
                    <Self as AutodiffBackend>::from_inner(hidden),
                )
            }
        }
    }
}

#[derive(Clone, Debug)]
struct GroupedSwiGluMlpState<B: GroupedLinearBackend> {
    input: FloatTensor<B>,
    in_weight: FloatTensor<B>,
    down_weight: FloatTensor<B>,
    counts: IntTensor<B>,
    projected: FloatTensor<B>,
    hidden: FloatTensor<B>,
    intermediate: usize,
    cache_key: Option<u64>,
}

#[derive(Debug)]
struct GroupedSwiGluMlpBackward;

impl<B: GroupedLinearBackend> Backward<B, 3> for GroupedSwiGluMlpBackward {
    type State = GroupedSwiGluMlpState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input, node_in_weight, node_down_weight] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let (grad_input, grad_in_weight, grad_down_weight) = B::grouped_swiglu_mlp_backward(
            state.input,
            state.in_weight,
            state.down_weight,
            state.counts,
            state.projected,
            state.hidden,
            grad,
            state.intermediate,
            state.cache_key,
        );
        if let Some(node) = node_input {
            grads.register::<B>(node.id, grad_input);
        }
        if let Some(node) = node_in_weight {
            grads.register::<B>(node.id, grad_in_weight);
        }
        if let Some(node) = node_down_weight {
            grads.register::<B>(node.id, grad_down_weight);
        }
    }
}

mod gpu {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use burn::backend::{TensorMetadata, ops::FloatTensorOps};
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::server::{Binding, GemmDescriptor, GemmMatrix, GroupedGemmDescriptor};

    use super::GroupedLinearBackend;
    use crate::model::cube_tensor::{empty_like, into_contiguous, zeros_like_dtype};
    use crate::model::fused_swiglu::FusedSwiGluBackend;

    const ZERO_THREADS: u32 = 256;

    fn count_cache() -> &'static Mutex<HashMap<u64, Vec<usize>>> {
        static CACHE: OnceLock<Mutex<HashMap<u64, Vec<usize>>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn cache_counts(key: u64, counts: &[usize]) {
        let mut cache = count_cache().lock().expect("MoE count cache poisoned");
        cache.insert(key, counts.to_vec());
        if cache.len() > 1024
            && let Some(oldest) = cache.keys().copied().min()
        {
            cache.remove(&oldest);
        }
    }

    fn take_cached_counts(key: u64) -> Option<Vec<usize>> {
        count_cache()
            .lock()
            .expect("MoE count cache poisoned")
            .remove(&key)
    }

    #[cube(launch)]
    fn fill_zero<F: Float>(output: &mut Tensor<F>) {
        if ABSOLUTE_POS < output.len() {
            output[ABSOLUTE_POS] = F::cast_from(0.0f32);
        }
    }

    fn contiguous_zeros<R: CubeRuntime>(like: &CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
        let output = empty_like(like, shape);
        let elements = output.meta.shape.num_elements() as u32;
        fill_zero::launch::<half::bf16, R>(
            &output.client,
            CubeCount::Static(elements.div_ceil(ZERO_THREADS), 1, 1),
            CubeDim::new_1d(ZERO_THREADS),
            output.clone().into_tensor_arg(),
        );
        output
    }

    fn sub_binding(
        base: &Binding,
        start_elements: usize,
        elements: usize,
        elem_size: usize,
    ) -> Binding {
        let start = (start_elements as u64)
            .checked_mul(elem_size as u64)
            .expect("grouped linear byte offset overflow");
        let length = (elements as u64)
            .checked_mul(elem_size as u64)
            .expect("grouped linear binding length overflow");
        let used = base.size_in_used();
        let end = start
            .checked_add(length)
            .expect("grouped linear binding end overflow");
        assert!(
            end <= used,
            "grouped linear view exceeds its tensor binding"
        );

        let mut binding = base.clone();
        binding.offset_start = Some(base.offset_start.unwrap_or(0) + start);
        binding.offset_end = Some(base.offset_end.unwrap_or(0) + used - end);
        binding
    }

    fn dimension(value: usize, name: &str) -> u32 {
        value
            .try_into()
            .unwrap_or_else(|_| panic!("grouped linear {name} exceeds u32::MAX"))
    }

    fn native_supported<R: CubeRuntime>(input: &CubeTensor<R>, weight: &CubeTensor<R>) -> bool {
        input.dtype == DType::BF16
            && weight.dtype == DType::BF16
            && input
                .client
                .features()
                .matmul
                .accelerated_grouped_gemm
                .contains(&ElemType::Float(FloatKind::BF16))
    }

    fn fallback_forward<R: CubeRuntime>(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        counts: &[usize],
    ) -> CubeTensor<R> {
        type B<R> = CubeBackend<R>;
        let [routes, input_width] = input.shape().dims();
        let [experts, weight_input, output_width] = weight.shape().dims();
        assert_eq!(counts.len(), experts);
        assert_eq!(counts.iter().sum::<usize>(), routes);
        assert_eq!(input_width, weight_input);

        let mut offset = 0;
        let mut outputs = Vec::with_capacity(experts);
        for (expert, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let input = B::<R>::float_slice(
                input.clone(),
                &[(offset..offset + count).into(), (0..input_width).into()],
            );
            let weight = B::<R>::float_reshape(
                B::<R>::float_slice(
                    weight.clone(),
                    &[
                        (expert..expert + 1).into(),
                        (0..input_width).into(),
                        (0..output_width).into(),
                    ],
                ),
                Shape::new([input_width, output_width]),
            );
            outputs.push(B::<R>::float_matmul(input, weight));
            offset += count;
        }
        assert_eq!(offset, routes);
        B::<R>::float_cat(outputs, 0)
    }

    fn native_forward<R: CubeRuntime>(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        counts: &[usize],
    ) -> CubeTensor<R> {
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let [routes, input_width] = input.shape().dims();
        let [experts, weight_input, output_width] = weight.shape().dims();
        assert_eq!(counts.len(), experts);
        assert_eq!(counts.iter().sum::<usize>(), routes);
        assert_eq!(input_width, weight_input);
        let output = empty_like(&input, Shape::new([routes, output_width]));
        let input_binding = input.handle.clone().binding();
        let weight_binding = weight.handle.clone().binding();
        let output_binding = output.handle.clone().binding();
        let elem_size = input.dtype.size();
        let elem = ElemType::Float(FloatKind::BF16);
        let mut offset = 0;
        let mut groups = Vec::with_capacity(experts);
        for (expert, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            groups.push(GemmDescriptor::new(
                GemmMatrix::new(
                    sub_binding(
                        &input_binding,
                        offset * input_width,
                        count * input_width,
                        elem_size,
                    ),
                    dimension(input_width, "input width"),
                    0,
                    false,
                ),
                GemmMatrix::new(
                    sub_binding(
                        &weight_binding,
                        expert * input_width * output_width,
                        input_width * output_width,
                        elem_size,
                    ),
                    dimension(output_width, "output width"),
                    0,
                    false,
                ),
                GemmMatrix::new(
                    sub_binding(
                        &output_binding,
                        offset * output_width,
                        count * output_width,
                        elem_size,
                    ),
                    dimension(output_width, "output width"),
                    0,
                    false,
                ),
                dimension(count, "route count"),
                dimension(output_width, "output width"),
                dimension(input_width, "input width"),
                1,
                elem,
            ));
            offset += count;
        }
        assert_eq!(offset, routes);
        input
            .client
            .grouped_gemm(GroupedGemmDescriptor::new(groups));
        output
    }

    fn fallback_backward<R: CubeRuntime>(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        grad: CubeTensor<R>,
        counts: &[usize],
    ) -> (CubeTensor<R>, CubeTensor<R>) {
        type B<R> = CubeBackend<R>;
        let [routes, input_width] = input.shape().dims();
        let [experts, _, output_width] = weight.shape().dims();
        assert_eq!(counts.len(), experts);
        assert_eq!(counts.iter().sum::<usize>(), routes);
        let zero_weights = zeros_like_dtype(&weight, weight.shape(), weight.dtype);
        let mut grad_inputs = Vec::with_capacity(experts);
        let mut grad_weights = Vec::with_capacity(experts);
        let mut offset = 0;
        for (expert, &count) in counts.iter().enumerate() {
            if count == 0 {
                grad_weights.push(B::<R>::float_slice(
                    zero_weights.clone(),
                    &[
                        (expert..expert + 1).into(),
                        (0..input_width).into(),
                        (0..output_width).into(),
                    ],
                ));
                continue;
            }
            let input_group = B::<R>::float_slice(
                input.clone(),
                &[(offset..offset + count).into(), (0..input_width).into()],
            );
            let grad_group = B::<R>::float_slice(
                grad.clone(),
                &[(offset..offset + count).into(), (0..output_width).into()],
            );
            let weight_group = B::<R>::float_reshape(
                B::<R>::float_slice(
                    weight.clone(),
                    &[
                        (expert..expert + 1).into(),
                        (0..input_width).into(),
                        (0..output_width).into(),
                    ],
                ),
                Shape::new([input_width, output_width]),
            );
            grad_inputs.push(B::<R>::float_matmul(
                grad_group.clone(),
                B::<R>::float_transpose(weight_group),
            ));
            grad_weights.push(B::<R>::float_reshape(
                B::<R>::float_matmul(B::<R>::float_transpose(input_group), grad_group),
                Shape::new([1, input_width, output_width]),
            ));
            offset += count;
        }
        assert_eq!(offset, routes);
        (
            B::<R>::float_cat(grad_inputs, 0),
            B::<R>::float_cat(grad_weights, 0),
        )
    }

    fn native_backward<R: CubeRuntime>(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        grad: CubeTensor<R>,
        counts: &[usize],
    ) -> (CubeTensor<R>, CubeTensor<R>) {
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let grad = into_contiguous(grad);
        let [routes, input_width] = input.shape().dims();
        let [experts, _, output_width] = weight.shape().dims();
        assert_eq!(counts.len(), experts);
        assert_eq!(counts.iter().sum::<usize>(), routes);
        let grad_input = empty_like(&input, Shape::new([routes, input_width]));
        let grad_weight = if counts.iter().all(|count| *count != 0) {
            empty_like(&weight, weight.shape())
        } else {
            contiguous_zeros(&weight, weight.shape())
        };
        let input_binding = input.handle.clone().binding();
        let weight_binding = weight.handle.clone().binding();
        let grad_binding = grad.handle.clone().binding();
        let grad_input_binding = grad_input.handle.clone().binding();
        let grad_weight_binding = grad_weight.handle.clone().binding();
        let elem_size = input.dtype.size();
        let elem = ElemType::Float(FloatKind::BF16);
        let active = counts.iter().filter(|count| **count != 0).count();
        let mut groups = Vec::with_capacity(active * 2);
        let mut offset = 0;

        for (expert, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let weight_start = expert * input_width * output_width;
            groups.push(GemmDescriptor::new(
                GemmMatrix::new(
                    sub_binding(
                        &grad_binding,
                        offset * output_width,
                        count * output_width,
                        elem_size,
                    ),
                    dimension(output_width, "output width"),
                    0,
                    false,
                ),
                GemmMatrix::new(
                    sub_binding(
                        &weight_binding,
                        weight_start,
                        input_width * output_width,
                        elem_size,
                    ),
                    dimension(output_width, "output width"),
                    0,
                    true,
                ),
                GemmMatrix::new(
                    sub_binding(
                        &grad_input_binding,
                        offset * input_width,
                        count * input_width,
                        elem_size,
                    ),
                    dimension(input_width, "input width"),
                    0,
                    false,
                ),
                dimension(count, "route count"),
                dimension(input_width, "input width"),
                dimension(output_width, "output width"),
                1,
                elem,
            ));
            offset += count;
        }

        offset = 0;
        for (expert, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let weight_start = expert * input_width * output_width;
            groups.push(GemmDescriptor::new(
                GemmMatrix::new(
                    sub_binding(
                        &input_binding,
                        offset * input_width,
                        count * input_width,
                        elem_size,
                    ),
                    dimension(input_width, "input width"),
                    0,
                    true,
                ),
                GemmMatrix::new(
                    sub_binding(
                        &grad_binding,
                        offset * output_width,
                        count * output_width,
                        elem_size,
                    ),
                    dimension(output_width, "output width"),
                    0,
                    false,
                ),
                GemmMatrix::new(
                    sub_binding(
                        &grad_weight_binding,
                        weight_start,
                        input_width * output_width,
                        elem_size,
                    ),
                    dimension(output_width, "output width"),
                    0,
                    false,
                ),
                dimension(input_width, "input width"),
                dimension(output_width, "output width"),
                dimension(count, "route count"),
                1,
                elem,
            ));
            offset += count;
        }
        assert_eq!(offset, routes);
        input
            .client
            .grouped_gemm(GroupedGemmDescriptor::new(groups));
        (grad_input, grad_weight)
    }

    fn read_counts<R: CubeRuntime>(counts: CubeTensor<R>, experts: usize) -> Vec<usize> {
        let counts = into_contiguous(counts);
        assert_eq!(counts.dtype, DType::I32, "MoE route counts must use i32");
        assert_eq!(counts.meta.shape.dims::<1>(), [experts]);
        let bytes = counts.client.read_one_unchecked(counts.handle);
        assert_eq!(bytes.len(), experts * size_of::<i32>());
        bytes
            .chunks_exact(size_of::<i32>())
            .map(|chunk| {
                let value = i32::from_ne_bytes(chunk.try_into().expect("i32 route-count bytes"));
                usize::try_from(value).expect("MoE route counts must be non-negative")
            })
            .collect()
    }

    fn grouped_forward<R: CubeRuntime>(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        counts: &[usize],
    ) -> CubeTensor<R> {
        if native_supported(&input, &weight) {
            native_forward(input, weight, counts)
        } else {
            fallback_forward(input, weight, counts)
        }
    }

    fn grouped_backward<R: CubeRuntime>(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        grad: CubeTensor<R>,
        counts: &[usize],
    ) -> (CubeTensor<R>, CubeTensor<R>) {
        if native_supported(&input, &weight) && grad.dtype == DType::BF16 {
            native_backward(input, weight, grad, counts)
        } else {
            fallback_backward(input, weight, grad, counts)
        }
    }

    impl<R: CubeRuntime> GroupedLinearBackend for CubeBackend<R> {
        fn grouped_linear_inner(
            input: CubeTensor<R>,
            weight: CubeTensor<R>,
            counts: Vec<usize>,
        ) -> CubeTensor<R> {
            grouped_forward(input, weight, &counts)
        }

        fn grouped_linear_backward(
            input: CubeTensor<R>,
            weight: CubeTensor<R>,
            grad: CubeTensor<R>,
            counts: Vec<usize>,
        ) -> (CubeTensor<R>, CubeTensor<R>) {
            grouped_backward(input, weight, grad, &counts)
        }

        fn grouped_swiglu_mlp_inner(
            input: CubeTensor<R>,
            in_weight: CubeTensor<R>,
            down_weight: CubeTensor<R>,
            counts: CubeTensor<R>,
            intermediate: usize,
            cache_key: Option<u64>,
        ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
            let experts = in_weight.meta.shape.dims::<3>()[0];
            let counts = read_counts(counts, experts);
            if let Some(cache_key) = cache_key {
                cache_counts(cache_key, &counts);
            }
            let projected = grouped_forward(input, in_weight, &counts);
            let hidden = <CubeBackend<R> as FusedSwiGluBackend>::fused_swiglu_inner(
                projected.clone(),
                intermediate,
            );
            let output = grouped_forward(hidden.clone(), down_weight, &counts);
            (output, projected, hidden)
        }

        fn grouped_swiglu_mlp_backward(
            input: CubeTensor<R>,
            in_weight: CubeTensor<R>,
            down_weight: CubeTensor<R>,
            counts: CubeTensor<R>,
            projected: CubeTensor<R>,
            hidden: CubeTensor<R>,
            grad: CubeTensor<R>,
            intermediate: usize,
            cache_key: Option<u64>,
        ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
            let experts = in_weight.meta.shape.dims::<3>()[0];
            let counts = cache_key
                .and_then(take_cached_counts)
                .unwrap_or_else(|| read_counts(counts, experts));
            let (grad_hidden, grad_down_weight) =
                grouped_backward(hidden, down_weight, grad, &counts);
            let grad_projected = <CubeBackend<R> as FusedSwiGluBackend>::fused_swiglu_backward(
                projected,
                grad_hidden,
                intermediate,
            );
            let (grad_input, grad_in_weight) =
                grouped_backward(input, in_weight, grad_projected, &counts);
            (grad_input, grad_in_weight, grad_down_weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{FloatDType, TensorData};

    use super::*;

    const COUNTS: [usize; 3] = [2, 0, 5];

    fn tensors(device: &Device) -> (Tensor<2>, Tensor<3>) {
        let input = (0..28)
            .map(|index| (index as f32 * 0.19).sin() * 0.4)
            .collect::<Vec<_>>();
        let weight = (0..60)
            .map(|index| (index as f32 * 0.13).cos() * 0.3)
            .collect::<Vec<_>>();
        (
            Tensor::<2>::from_data(TensorData::new(input, [7, 4]), device)
                .cast(FloatDType::BF16)
                .require_grad(),
            Tensor::<3>::from_data(TensorData::new(weight, [3, 4, 5]), device)
                .cast(FloatDType::BF16)
                .require_grad(),
        )
    }

    fn factors(device: &Device) -> Tensor<2> {
        Tensor::<2>::from_data(
            TensorData::new(
                (0..35)
                    .map(|index| 0.2 + (index as f32 * 0.07).sin())
                    .collect::<Vec<_>>(),
                [7, 5],
            ),
            device,
        )
        .cast(FloatDType::BF16)
    }

    fn reference(input: Tensor<2>, weight: Tensor<3>) -> Tensor<2> {
        let mut offset = 0;
        let mut outputs = Vec::new();
        for (expert, &count) in COUNTS.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let input_group = input.clone().slice([offset..offset + count, 0..4]);
            let weight_group = weight
                .clone()
                .slice([expert..expert + 1, 0..4, 0..5])
                .squeeze_dim::<2>(0);
            outputs.push(input_group.matmul(weight_group));
            offset += count;
        }
        Tensor::cat(outputs, 0)
    }

    fn run(grouped: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let device = crate::model::default_device().autodiff();
        let (input, weight) = tensors(&device);
        let output = if grouped {
            grouped_linear(input.clone(), weight.clone(), &COUNTS)
        } else {
            reference(input.clone(), weight.clone())
        };
        let output_values = output
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let mut gradients = (output * factors(&device)).sum().backward();
        let grad_input = input
            .grad_remove(&mut gradients)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let grad_weight = weight
            .grad_remove(&mut gradients)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        (output_values, grad_input, grad_weight)
    }

    #[test]
    fn grouped_forward_and_backward_match_tensor_ops() {
        let actual = run(true);
        let expected = run(false);
        for (name, actual, expected) in [
            ("output", actual.0, expected.0),
            ("input gradient", actual.1, expected.1),
            ("weight gradient", actual.2, expected.2),
        ] {
            let (index, difference) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
                .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
                .unwrap();
            assert!(
                difference <= 0.03,
                "{name} max difference {difference} at {index}: {} != {}",
                actual[index],
                expected[index]
            );
        }
    }
}
