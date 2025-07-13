//! Fused depthwise Conv1d used by Mamba blocks.
//!
//! Burn's generic grouped-convolution weight gradient launches one convolution
//! per group. Mamba has one group per channel, so GPU training uses dedicated
//! CubeCL kernels while CPU keeps Burn's reference implementation.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use burn::backend::NdArray;
use burn::backend::{
    Backend, Dispatch, TensorMetadata, backend_extension, ops::ConvOptions, tensor::FloatTensor,
};
use burn::prelude::*;
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
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
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
        let bias_grad = Self::float_sum_dim(grad, 0);
        let bias_grad = Self::float_sum_dim(bias_grad, 2);
        let bias_grad = Self::float_reshape(bias_grad, [channels].into());
        (input_grad, weight_grad, bias_grad)
    }
}

pub(super) fn depthwise_conv1d(input: Tensor<3>, weight: Tensor<3>, bias: Tensor<1>) -> Tensor<3> {
    if input.device() == Device::ndarray() {
        let channels = input.dims()[1];
        return burn::tensor::module::conv1d(
            input,
            weight,
            Some(bias),
            ConvOptions::new([1], [0], [1], channels),
        );
    }
    let output = Dispatch::depthwise_conv1d_inner(
        input.into_dispatch(),
        weight.into_dispatch(),
        bias.into_dispatch(),
    );
    Tensor::from_dispatch(output)
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
            (node_input, output.0),
            (node_weight, output.1),
            (node_bias, output.2),
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
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};

    use super::DepthwiseConv1dBackend;
    use crate::model::cube_tensor::{empty_like, into_contiguous};

    const ELEMENTWISE_THREADS: u32 = 256;
    const REDUCTION_THREADS: u32 = 32;

    #[cube(launch)]
    fn depthwise_conv1d_forward(
        input: &Tensor<f32>,
        weight: &Tensor<f32>,
        bias: &Tensor<f32>,
        output: &mut Tensor<f32>,
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
        weight: &Tensor<f32>,
        grad: &Tensor<f32>,
        grad_input: &mut Tensor<f32>,
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
        input: &Tensor<f32>,
        grad: &Tensor<f32>,
        grad_weight: &mut Tensor<f32>,
        grad_bias: &mut Tensor<f32>,
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

    impl<R: CubeRuntime> DepthwiseConv1dBackend for CubeBackend<R> {
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
                input.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.into_tensor_arg(),
                output.clone().into_tensor_arg(),
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
        ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
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
                weight.into_tensor_arg(),
                grad.clone().into_tensor_arg(),
                grad_input.clone().into_tensor_arg(),
                channels as u32,
                input_len as u32,
                output_len as u32,
                kernel_size,
            );
            depthwise_conv1d_backward_params::launch::<R>(
                &client,
                CubeCount::Static((channels * kernel_size) as u32, 1, 1),
                CubeDim::new_1d(REDUCTION_THREADS),
                input.into_tensor_arg(),
                grad.into_tensor_arg(),
                grad_weight.clone().into_tensor_arg(),
                grad_bias.clone().into_tensor_arg(),
                batch as u32,
                channels as u32,
                input_len as u32,
                output_len as u32,
                kernel_size,
            );

            (grad_input, grad_weight, grad_bias)
        }
    }
}

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::depthwise_conv1d;
    use crate::model::test_support::{max_diff, snapshot, values};

    #[test]
    fn gpu_depthwise_conv1d_matches_ndarray_forward_and_backward() {
        let cpu = Device::ndarray().autodiff();
        let gpu = crate::model::default_device().autodiff();
        let (batch, channels, input_len, kernel_size) = (2, 3, 7, 3);
        let output_len = input_len - kernel_size + 1;
        let input_data = values(batch * channels * input_len, 0.13, 0.0);
        let weight_data = values(channels * kernel_size, 0.19, 0.0);
        let bias_data = values(channels, 0.23, 0.0);
        let output_weights = values(batch * channels * output_len, 0.29, 0.0);

        macro_rules! run {
            ($device:expr) => {{
                let input = Tensor::<3>::from_data(
                    TensorData::new(input_data.clone(), [batch, channels, input_len]),
                    $device,
                )
                .require_grad();
                let weight = Tensor::<3>::from_data(
                    TensorData::new(weight_data.clone(), [channels, 1, kernel_size]),
                    $device,
                )
                .require_grad();
                let bias =
                    Tensor::<1>::from_data(TensorData::new(bias_data.clone(), [channels]), $device)
                        .require_grad();
                let output = depthwise_conv1d(input.clone(), weight.clone(), bias.clone());
                let output_data = snapshot(output.clone());
                let factors = Tensor::<3>::from_data(
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

        let cpu = run!(&cpu);
        let gpu = run!(&gpu);
        for (name, cpu, gpu, tolerance) in [
            ("output", cpu.0, gpu.0, 1e-5),
            ("input", cpu.1, gpu.1, 1e-5),
            ("weight", cpu.2, gpu.2, 1e-4),
            ("bias", cpu.3, gpu.3, 1e-5),
        ] {
            let difference = max_diff(cpu, gpu);
            assert!(
                difference < tolerance,
                "{name} max diff {difference} exceeds {tolerance}"
            );
        }
    }
}
