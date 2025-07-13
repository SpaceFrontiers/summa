//! Fused depthwise Conv1d used by Mamba blocks.
//!
//! Burn's generic grouped-convolution weight gradient launches one convolution
//! per group. Mamba has one group per channel, so GPU training uses dedicated
//! CubeCL kernels while CPU keeps Burn's reference implementation.

use burn::prelude::*;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{TensorMetadata, TensorPrimitive, ops::FloatTensor};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DepthwiseConv1dGradients<B: Backend> {
    input: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
}

pub trait DepthwiseConv1dBackend: Backend {
    fn depthwise_conv1d_inner(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let channels = input.shape()[1];
        Self::conv1d(
            input,
            weight,
            Some(bias),
            ConvOptions::new([1], [0], [1], channels),
        )
    }

    fn depthwise_conv1d_backward(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> DepthwiseConv1dGradients<Self> {
        let channels = input.shape()[1];
        let input_grad = Self::conv1d_x_backward(
            input.clone(),
            weight.clone(),
            grad.clone(),
            ConvOptions::new([1], [0], [1], channels),
        );
        let weight_grad = Self::conv1d_weight_backward(
            input,
            weight,
            grad.clone(),
            ConvOptions::new([1], [0], [1], channels),
        );
        let bias_grad = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(grad))
            .sum_dim(0)
            .sum_dim(2)
            .reshape([channels])
            .into_primitive()
            .tensor();
        DepthwiseConv1dGradients {
            input: input_grad,
            weight: weight_grad,
            bias: bias_grad,
        }
    }
}

pub(super) fn depthwise_conv1d<B: DepthwiseConv1dBackend>(
    input: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let output = B::depthwise_conv1d_inner(
        input.into_primitive().tensor(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

impl DepthwiseConv1dBackend for burn_ndarray::NdArray {}

#[derive(Clone, Debug)]
struct DepthwiseConv1dState<B: DepthwiseConv1dBackend> {
    input: FloatTensor<B>,
    weight: FloatTensor<B>,
}

#[derive(Debug)]
struct DepthwiseConv1dBackward;

impl<B: DepthwiseConv1dBackend> Backward<B, 3> for DepthwiseConv1dBackward {
    type State = DepthwiseConv1dState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_input, node_weight, node_bias] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let output = B::depthwise_conv1d_backward(ops.state.input, ops.state.weight, grad);
        for (node, grad) in [
            (node_input, output.input),
            (node_weight, output.weight),
            (node_bias, output.bias),
        ] {
            if let Some(node) = node {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

impl<B: DepthwiseConv1dBackend, C: CheckpointStrategy> DepthwiseConv1dBackend for Autodiff<B, C> {
    fn depthwise_conv1d_inner(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        match DepthwiseConv1dBackward
            .prepare::<C>([input.node.clone(), weight.node.clone(), bias.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::depthwise_conv1d_inner(
                    input.primitive.clone(),
                    weight.primitive.clone(),
                    bias.primitive,
                );
                let state = DepthwiseConv1dState {
                    input: input.primitive,
                    weight: weight.primitive,
                };
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::depthwise_conv1d_inner(
                input.primitive,
                weight.primitive,
                bias.primitive,
            )),
        }
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use burn::tensor::Shape;
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::element::{BoolElement, IntElement};
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};

    use super::{DepthwiseConv1dBackend, DepthwiseConv1dGradients};
    use crate::burn_model::cube_tensor::{empty_like, into_contiguous};

    const ELEMENTWISE_THREADS: u32 = 256;
    const REDUCTION_THREADS: u32 = 32;

    #[cube(launch)]
    fn depthwise_conv1d_forward(
        input: &Array<f32>,
        weight: &Array<f32>,
        bias: &Array<f32>,
        output: &mut Array<f32>,
        channels: u32,
        input_len: u32,
        output_len: u32,
        #[comptime] kernel_size: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx < output.len() {
            let output_len = output_len as usize;
            let input_len = input_len as usize;
            let channels = channels as usize;
            let t = idx % output_len;
            let channel_batch = idx / output_len;
            let channel = channel_batch % channels;
            let input_base = channel_batch * input_len + t;
            let weight_base = channel * kernel_size;
            let mut value = bias[channel];
            for k in 0..kernel_size {
                value += input[input_base + k] * weight[weight_base + k];
            }
            output[idx] = value;
        }
    }

    #[cube(launch)]
    fn depthwise_conv1d_backward_input(
        weight: &Array<f32>,
        grad: &Array<f32>,
        grad_input: &mut Array<f32>,
        channels: u32,
        input_len: u32,
        output_len: u32,
        #[comptime] kernel_size: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx < grad_input.len() {
            let input_len = input_len as usize;
            let output_len = output_len as usize;
            let channels = channels as usize;
            let input_t = idx % input_len;
            let channel_batch = idx / input_len;
            let channel = channel_batch % channels;
            let grad_base = channel_batch * output_len;
            let weight_base = channel * kernel_size;
            let mut value = 0.0f32;
            for k in 0..kernel_size {
                if input_t >= k {
                    let t = input_t - k;
                    if t < output_len {
                        value += grad[grad_base + t] * weight[weight_base + k];
                    }
                }
            }
            grad_input[idx] = value;
        }
    }

    #[cube(launch)]
    fn depthwise_conv1d_backward_params(
        input: &Array<f32>,
        grad: &Array<f32>,
        grad_weight: &mut Array<f32>,
        grad_bias: &mut Array<f32>,
        batch: u32,
        channels: u32,
        input_len: u32,
        output_len: u32,
        #[comptime] kernel_size: usize,
    ) {
        let param = CUBE_POS_X as usize;
        let channel = param / kernel_size;
        let k = param % kernel_size;
        let lane = UNIT_POS_X as usize;
        let batch = batch as usize;
        let channels = channels as usize;
        let input_len = input_len as usize;
        let output_len = output_len as usize;
        let sample_count = batch * output_len;
        let mut weight_sum = 0.0f32;
        let mut bias_sum = 0.0f32;
        let mut sample = lane;
        while sample < sample_count {
            let batch_idx = sample / output_len;
            let t = sample % output_len;
            let grad_idx = (batch_idx * channels + channel) * output_len + t;
            let grad_value = grad[grad_idx];
            let input_idx = (batch_idx * channels + channel) * input_len + t + k;
            weight_sum += grad_value * input[input_idx];
            if k == 0 {
                bias_sum += grad_value;
            }
            sample += REDUCTION_THREADS as usize;
        }
        let weight_sum = plane_sum(weight_sum);
        let bias_sum = plane_sum(bias_sum);
        if lane == 0 {
            grad_weight[param] = weight_sum;
            if k == 0 {
                grad_bias[channel] = bias_sum;
            }
        }
    }

    impl<R, I, BT> DepthwiseConv1dBackend for CubeBackend<R, f32, I, BT>
    where
        R: CubeRuntime,
        I: IntElement,
        BT: BoolElement,
    {
        fn depthwise_conv1d_inner(
            input: CubeTensor<R>,
            weight: CubeTensor<R>,
            bias: CubeTensor<R>,
        ) -> CubeTensor<R> {
            let [batch, channels, input_len] = input.meta.shape.dims();
            let [weight_channels, group_channels, kernel_size] = weight.meta.shape.dims();
            assert_eq!(weight_channels, channels);
            assert_eq!(group_channels, 1);
            assert!(input_len >= kernel_size);
            let output_len = input_len - kernel_size + 1;
            let input = into_contiguous(input);
            let weight = into_contiguous(weight);
            let bias = into_contiguous(bias);
            let output = empty_like(&input, Shape::new([batch, channels, output_len]));
            let total = (batch * channels * output_len) as u32;
            let client = input.client.clone();
            depthwise_conv1d_forward::launch::<R>(
                &client,
                CubeCount::Static(total.div_ceil(ELEMENTWISE_THREADS), 1, 1),
                CubeDim::new_1d(ELEMENTWISE_THREADS),
                input.into_array_arg(),
                weight.into_array_arg(),
                bias.into_array_arg(),
                output.clone().into_array_arg(),
                channels as u32,
                input_len as u32,
                output_len as u32,
                kernel_size,
            );
            output
        }

        fn depthwise_conv1d_backward(
            input: CubeTensor<R>,
            weight: CubeTensor<R>,
            grad: CubeTensor<R>,
        ) -> DepthwiseConv1dGradients<Self> {
            let [batch, channels, input_len] = input.meta.shape.dims();
            let [_, _, kernel_size] = weight.meta.shape.dims();
            let [_, _, output_len] = grad.meta.shape.dims();
            assert_eq!(output_len, input_len - kernel_size + 1);
            let input = into_contiguous(input);
            let weight = into_contiguous(weight);
            let grad = into_contiguous(grad);
            let grad_input = empty_like(&input, Shape::new([batch, channels, input_len]));
            let grad_weight = empty_like(&weight, Shape::new([channels, 1, kernel_size]));
            let grad_bias = empty_like(&weight, Shape::new([channels]));
            let client = input.client.clone();

            let input_total = (batch * channels * input_len) as u32;
            depthwise_conv1d_backward_input::launch::<R>(
                &client,
                CubeCount::Static(input_total.div_ceil(ELEMENTWISE_THREADS), 1, 1),
                CubeDim::new_1d(ELEMENTWISE_THREADS),
                weight.into_array_arg(),
                grad.clone().into_array_arg(),
                grad_input.clone().into_array_arg(),
                channels as u32,
                input_len as u32,
                output_len as u32,
                kernel_size,
            );
            depthwise_conv1d_backward_params::launch::<R>(
                &client,
                CubeCount::Static((channels * kernel_size) as u32, 1, 1),
                CubeDim::new_1d(REDUCTION_THREADS),
                input.into_array_arg(),
                grad.into_array_arg(),
                grad_weight.clone().into_array_arg(),
                grad_bias.clone().into_array_arg(),
                batch as u32,
                channels as u32,
                input_len as u32,
                output_len as u32,
                kernel_size,
            );

            DepthwiseConv1dGradients {
                input: grad_input,
                weight: grad_weight,
                bias: grad_bias,
            }
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use burn::tensor::{Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use burn_wgpu::Wgpu;

    use super::depthwise_conv1d;
    use crate::burn_model::test_support::{max_diff, values};

    #[test]
    fn fused_depthwise_conv1d_matches_ndarray_forward_and_backward() {
        type Cpu = Autodiff<NdArray>;
        type Gpu = Autodiff<Wgpu>;
        let cpu = Default::default();
        let gpu = Default::default();
        let (batch, channels, input_len, kernel_size) = (2, 3, 7, 3);
        let output_len = input_len - kernel_size + 1;
        let input_data = values(batch * channels * input_len, 0.13, 0.0);
        let weight_data = values(channels * kernel_size, 0.19, 0.0);
        let bias_data = values(channels, 0.23, 0.0);
        let output_weights = values(batch * channels * output_len, 0.29, 0.0);

        macro_rules! run {
            ($backend:ty, $device:expr) => {{
                let input = Tensor::<$backend, 3>::from_data(
                    TensorData::new(input_data.clone(), [batch, channels, input_len]),
                    $device,
                )
                .require_grad();
                let weight = Tensor::<$backend, 3>::from_data(
                    TensorData::new(weight_data.clone(), [channels, 1, kernel_size]),
                    $device,
                )
                .require_grad();
                let bias = Tensor::<$backend, 1>::from_data(
                    TensorData::new(bias_data.clone(), [channels]),
                    $device,
                )
                .require_grad();
                let output = depthwise_conv1d(input.clone(), weight.clone(), bias.clone());
                let output_data = output.clone().inner().into_data();
                let factors = Tensor::<$backend, 3>::from_data(
                    TensorData::new(output_weights.clone(), [batch, channels, output_len]),
                    $device,
                );
                let mut grads = (output * factors).sum().backward();
                (
                    output_data,
                    input.grad_remove(&mut grads).unwrap().into_data(),
                    weight.grad_remove(&mut grads).unwrap().into_data(),
                    bias.grad_remove(&mut grads).unwrap().into_data(),
                )
            }};
        }

        let cpu = run!(Cpu, &cpu);
        let gpu = run!(Gpu, &gpu);
        assert!(max_diff(cpu.0, gpu.0) < 1e-5);
        assert!(max_diff(cpu.1, gpu.1) < 1e-5);
        assert!(max_diff(cpu.2, gpu.2) < 1e-4);
        assert!(max_diff(cpu.3, gpu.3) < 1e-5);
    }
}
