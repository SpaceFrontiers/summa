//! Lazy-fusion adapters for Hermes' custom CUDA operations.
//!
//! Burn can fuse the ordinary elementwise and reduction graph around these
//! operations. The adapters record custom kernels as opaque operations while
//! leaving attention's recompute backward visible to Burn's fusion planner.

use std::marker::PhantomData;

use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, Shape};
use burn_cubecl::{CubeBackend, fusion::FusionCubeRuntime};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, ScalarIr, TensorIr};
use cubecl::cuda::CudaRuntime;

use super::conv::DepthwiseConv1dBackend;
use super::fused_attention::{AttentionBackend, chunked_attention_backward};
use super::linear_cross_entropy::LinearCrossEntropyBackend;
use super::scan::{MambaBackend, scan_checkpoint_interval};

const ATTENTION_BACKWARD_CHUNK_ROWS: usize = 2048;

#[derive(Debug)]
struct DepthwiseForward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B> Operation<B::FusionRuntime> for DepthwiseForward<B>
where
    B: FusionBackend + DepthwiseConv1dBackend,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([input, weight, bias], [output_ir]) = self.desc.as_fixed();
        let output = B::depthwise_conv1d_inner(
            handles.get_float_tensor::<B>(input),
            handles.get_float_tensor::<B>(weight),
            handles.get_float_tensor::<B>(bias),
        );
        handles.register_float_tensor::<B>(&output_ir.id, output);
    }
}

#[derive(Debug)]
struct DepthwiseBackward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B> Operation<B::FusionRuntime> for DepthwiseBackward<B>
where
    B: FusionBackend + DepthwiseConv1dBackend,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([input, weight, grad], [grad_input, grad_weight, grad_bias]) = self.desc.as_fixed();
        let (input, weight, bias) = B::depthwise_conv1d_backward(
            handles.get_float_tensor::<B>(input),
            handles.get_float_tensor::<B>(weight),
            handles.get_float_tensor::<B>(grad),
        );
        handles.register_float_tensor::<B>(&grad_input.id, input);
        handles.register_float_tensor::<B>(&grad_weight.id, weight);
        handles.register_float_tensor::<B>(&grad_bias.id, bias);
    }
}

impl<B> DepthwiseConv1dBackend for Fusion<B>
where
    B: FusionBackend + DepthwiseConv1dBackend,
{
    fn depthwise_conv1d_inner(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let [batch, channels, input_len] = input.shape.dims();
        let kernel_size = weight.shape.dims::<3>()[2];
        let client = input.client.clone();
        let output = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, channels, input_len - kernel_size + 1]),
            input.dtype,
        );
        let desc = CustomOpIr::new(
            "hermes_depthwise_conv1d",
            &[input.into_ir(), weight.into_ir(), bias.into_ir()],
            &[output],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                DepthwiseForward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .remove(0)
    }

    fn depthwise_conv1d_backward(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let channels = input.shape.dims::<3>()[1];
        let client = input.client.clone();
        let grad_input = TensorIr::uninit(
            client.create_empty_handle(),
            input.shape.clone(),
            input.dtype,
        );
        let grad_weight = TensorIr::uninit(
            client.create_empty_handle(),
            weight.shape.clone(),
            weight.dtype,
        );
        let grad_bias = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([channels]),
            weight.dtype,
        );
        let desc = CustomOpIr::new(
            "hermes_depthwise_conv1d_backward",
            &[input.into_ir(), weight.into_ir(), grad.into_ir()],
            &[grad_input, grad_weight, grad_bias],
        );
        let [input, weight, bias] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                DepthwiseBackward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .try_into()
            .expect("depthwise backward has three outputs");
        (input, weight, bias)
    }
}

#[derive(Debug)]
struct LinearCrossEntropyForward<B> {
    desc: CustomOpIr,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
    backend: PhantomData<B>,
}

impl<B> Operation<B::FusionRuntime> for LinearCrossEntropyForward<B>
where
    B: FusionBackend + LinearCrossEntropyBackend,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([hidden, weight, bias, targets], [loss_ir]) = self.desc.as_fixed();
        let loss = B::linear_cross_entropy_inner(
            handles.get_float_tensor::<B>(hidden),
            handles.get_float_tensor::<B>(weight),
            handles.get_float_tensor::<B>(bias),
            handles.get_int_tensor::<B>(targets),
            self.logical_vocab_size,
            self.chunk_size,
            self.use_bias,
        );
        handles.register_float_tensor::<B>(&loss_ir.id, loss);
    }
}

#[derive(Debug)]
struct LinearCrossEntropyBackward<B> {
    desc: CustomOpIr,
    logical_vocab_size: usize,
    chunk_size: usize,
    use_bias: bool,
    backend: PhantomData<B>,
}

impl<B> Operation<B::FusionRuntime> for LinearCrossEntropyBackward<B>
where
    B: FusionBackend + LinearCrossEntropyBackend,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([hidden, weight, bias, targets, grad_output], [grad_hidden, grad_weight, grad_bias]) =
            self.desc.as_fixed();
        let output = B::linear_cross_entropy_backward(
            handles.get_float_tensor::<B>(hidden),
            handles.get_float_tensor::<B>(weight),
            handles.get_float_tensor::<B>(bias),
            handles.get_int_tensor::<B>(targets),
            handles.get_float_tensor::<B>(grad_output),
            self.logical_vocab_size,
            self.chunk_size,
            self.use_bias,
        );
        handles.register_float_tensor::<B>(&grad_hidden.id, output.0);
        handles.register_float_tensor::<B>(&grad_weight.id, output.1);
        handles.register_float_tensor::<B>(&grad_bias.id, output.2);
    }
}

impl<B> LinearCrossEntropyBackend for Fusion<B>
where
    B: FusionBackend + LinearCrossEntropyBackend,
{
    fn linear_cross_entropy_inner(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: burn::backend::tensor::IntTensor<Self>,
        logical_vocab_size: usize,
        chunk_size: usize,
        use_bias: bool,
    ) -> FloatTensor<Self> {
        let client = hidden.client.clone();
        // The loss statistics chain always computes in FP32, even when the
        // hidden activations arrive in the BF16 residual-stream dtype.
        let loss = TensorIr::uninit(client.create_empty_handle(), Shape::new([1]), DType::F32);
        let desc = CustomOpIr::with_scalars(
            "hermes_linear_cross_entropy",
            &[
                hidden.into_ir(),
                weight.into_ir(),
                bias.into_ir(),
                targets.into_ir(),
            ],
            &[loss],
            vec![
                ScalarIr::UInt(logical_vocab_size as u64),
                ScalarIr::UInt(chunk_size as u64),
                ScalarIr::Bool(use_bias),
            ],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                LinearCrossEntropyForward::<B> {
                    desc,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                    backend: PhantomData,
                },
            )
            .remove(0)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_cross_entropy_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: burn::backend::tensor::IntTensor<Self>,
        grad_output: FloatTensor<Self>,
        logical_vocab_size: usize,
        chunk_size: usize,
        use_bias: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let client = hidden.client.clone();
        let grad_hidden = TensorIr::uninit(
            client.create_empty_handle(),
            hidden.shape.clone(),
            hidden.dtype,
        );
        let grad_weight = TensorIr::uninit(
            client.create_empty_handle(),
            weight.shape.clone(),
            weight.dtype,
        );
        let grad_bias =
            TensorIr::uninit(client.create_empty_handle(), bias.shape.clone(), bias.dtype);
        let desc = CustomOpIr::with_scalars(
            "hermes_linear_cross_entropy_backward",
            &[
                hidden.into_ir(),
                weight.into_ir(),
                bias.into_ir(),
                targets.into_ir(),
                grad_output.into_ir(),
            ],
            &[grad_hidden, grad_weight, grad_bias],
            vec![
                ScalarIr::UInt(logical_vocab_size as u64),
                ScalarIr::UInt(chunk_size as u64),
                ScalarIr::Bool(use_bias),
            ],
        );
        let [grad_hidden, grad_weight, grad_bias] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                LinearCrossEntropyBackward::<B> {
                    desc,
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                    backend: PhantomData,
                },
            )
            .try_into()
            .expect("linear cross-entropy backward has three outputs");
        (grad_hidden, grad_weight, grad_bias)
    }
}

#[derive(Debug)]
struct ScanForward<B> {
    desc: CustomOpIr,
    state_dim: usize,
    save_states: bool,
    backend: PhantomData<B>,
}

impl<B> Operation<B::FusionRuntime> for ScanForward<B>
where
    B: FusionBackend + MambaBackend,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([delta, xs, b_mat, c_mat, a, d, h], [y_ir, h_ir, states_ir]) = self.desc.as_fixed();
        let (y, h, states) = B::selective_scan_inner(
            handles.get_float_tensor::<B>(delta),
            handles.get_float_tensor::<B>(xs),
            handles.get_float_tensor::<B>(b_mat),
            handles.get_float_tensor::<B>(c_mat),
            handles.get_float_tensor::<B>(a),
            handles.get_float_tensor::<B>(d),
            handles.get_float_tensor::<B>(h),
            self.state_dim,
            self.save_states,
        );
        handles.register_float_tensor::<B>(&y_ir.id, y);
        handles.register_float_tensor::<B>(&h_ir.id, h);
        handles.register_float_tensor::<B>(&states_ir.id, states);
    }
}

#[derive(Debug)]
struct ScanBackward<B> {
    desc: CustomOpIr,
    state_dim: usize,
    backend: PhantomData<B>,
}

impl<B> Operation<B::FusionRuntime> for ScanBackward<B>
where
    B: FusionBackend + MambaBackend,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let (
            [delta, xs, b_mat, c_mat, a, d, h, states, grad_y],
            [grad_delta, grad_xs, grad_b, grad_c, grad_a, grad_d, grad_h],
        ) = self.desc.as_fixed();
        let output = B::selective_scan_backward(
            handles.get_float_tensor::<B>(delta),
            handles.get_float_tensor::<B>(xs),
            handles.get_float_tensor::<B>(b_mat),
            handles.get_float_tensor::<B>(c_mat),
            handles.get_float_tensor::<B>(a),
            handles.get_float_tensor::<B>(d),
            handles.get_float_tensor::<B>(h),
            handles.get_float_tensor::<B>(states),
            handles.get_float_tensor::<B>(grad_y),
            self.state_dim,
        );
        handles.register_float_tensor::<B>(&grad_delta.id, output.0);
        handles.register_float_tensor::<B>(&grad_xs.id, output.1);
        handles.register_float_tensor::<B>(&grad_b.id, output.2);
        handles.register_float_tensor::<B>(&grad_c.id, output.3);
        handles.register_float_tensor::<B>(&grad_a.id, output.4);
        handles.register_float_tensor::<B>(&grad_d.id, output.5);
        handles.register_float_tensor::<B>(&grad_h.id, output.6);
    }
}

impl<B> MambaBackend for Fusion<B>
where
    B: FusionBackend + MambaBackend,
{
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
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let [batch, seq_len, channels] = xs.shape.dims();
        let client = xs.client.clone();
        let dtype = xs.dtype;
        let checkpoint_interval = scan_checkpoint_interval(batch, seq_len, channels, state_dim);
        let y = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, seq_len, channels]),
            dtype,
        );
        let h_out = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, channels, state_dim]),
            DType::F32,
        );
        let states_shape = if save_states {
            Shape::new([
                batch,
                seq_len.div_ceil(checkpoint_interval),
                channels,
                state_dim,
            ])
        } else {
            Shape::new([batch, channels, state_dim])
        };
        let states = TensorIr::uninit(client.create_empty_handle(), states_shape, DType::F32);
        let desc = CustomOpIr::with_scalars(
            "hermes_selective_scan",
            &[
                delta.into_ir(),
                xs.into_ir(),
                b_mat.into_ir(),
                c_mat.into_ir(),
                a.into_ir(),
                d.into_ir(),
                h.into_ir(),
            ],
            &[y, h_out, states],
            vec![
                ScalarIr::UInt(state_dim as u64),
                ScalarIr::Bool(save_states),
            ],
        );
        let [y, h, states] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                ScanForward::<B> {
                    desc,
                    state_dim,
                    save_states,
                    backend: PhantomData,
                },
            )
            .try_into()
            .expect("selective scan has three outputs");
        (y, h, states)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
        let [batch, seq_len, channels] = xs.shape.dims();
        let client = xs.client.clone();
        let dtype = xs.dtype;
        let delta_shape = Shape::new([batch, seq_len, channels]);
        let bc_shape = Shape::new([batch, seq_len, state_dim]);
        let a_shape = Shape::new([channels, state_dim]);
        let d_shape = Shape::new([channels]);
        let h_shape = Shape::new([batch, channels, state_dim]);
        let outputs = [
            TensorIr::uninit(client.create_empty_handle(), delta_shape.clone(), dtype),
            TensorIr::uninit(client.create_empty_handle(), delta_shape, dtype),
            TensorIr::uninit(client.create_empty_handle(), bc_shape.clone(), dtype),
            TensorIr::uninit(client.create_empty_handle(), bc_shape, dtype),
            TensorIr::uninit(client.create_empty_handle(), a_shape, DType::F32),
            TensorIr::uninit(client.create_empty_handle(), d_shape, DType::F32),
            TensorIr::uninit(client.create_empty_handle(), h_shape, DType::F32),
        ];
        let desc = CustomOpIr::with_scalars(
            "hermes_selective_scan_backward",
            &[
                delta.into_ir(),
                xs.into_ir(),
                b_mat.into_ir(),
                c_mat.into_ir(),
                a.into_ir(),
                d.into_ir(),
                h.into_ir(),
                states.into_ir(),
                grad_y.into_ir(),
            ],
            &outputs,
            vec![ScalarIr::UInt(state_dim as u64)],
        );
        let [grad_delta, grad_xs, grad_b, grad_c, grad_a, grad_d, grad_h] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                ScanBackward::<B> {
                    desc,
                    state_dim,
                    backend: PhantomData,
                },
            )
            .try_into()
            .expect("selective scan backward has seven outputs");
        (grad_delta, grad_xs, grad_b, grad_c, grad_a, grad_d, grad_h)
    }
}

#[derive(Debug)]
struct AttentionForward {
    desc: CustomOpIr,
    causal: bool,
}

#[derive(Debug)]
struct AttentionCausalMask {
    desc: CustomOpIr,
    row_offset: usize,
}

impl Operation<FusionCubeRuntime<CudaRuntime>> for AttentionCausalMask {
    fn execute(
        &self,
        handles: &mut HandleContainer<
            <FusionCubeRuntime<CudaRuntime> as FusionRuntime>::FusionHandle,
        >,
    ) {
        type B = CubeBackend<CudaRuntime>;
        let ([scores], [output]) = self.desc.as_fixed();
        let result =
            B::attention_causal_mask(handles.get_float_tensor::<B>(scores), self.row_offset);
        handles.register_float_tensor::<B>(&output.id, result);
    }
}

impl Operation<FusionCubeRuntime<CudaRuntime>> for AttentionForward {
    fn execute(
        &self,
        handles: &mut HandleContainer<
            <FusionCubeRuntime<CudaRuntime> as FusionRuntime>::FusionHandle,
        >,
    ) {
        type B = CubeBackend<CudaRuntime>;
        let ([query, key, value], [output, saved_query, saved_key, saved_value, saved_output]) =
            self.desc.as_fixed();
        let result = B::attention_inner(
            handles.get_float_tensor::<B>(query),
            handles.get_float_tensor::<B>(key),
            handles.get_float_tensor::<B>(value),
            self.causal,
        );
        handles.register_float_tensor::<B>(&output.id, result.0);
        handles.register_float_tensor::<B>(&saved_query.id, result.1);
        handles.register_float_tensor::<B>(&saved_key.id, result.2);
        handles.register_float_tensor::<B>(&saved_value.id, result.3);
        handles.register_float_tensor::<B>(&saved_output.id, result.4);
    }
}

#[derive(Debug)]
struct AttentionProbabilities {
    desc: CustomOpIr,
    scale: f32,
    row_offset: usize,
    causal: bool,
}

impl Operation<FusionCubeRuntime<CudaRuntime>> for AttentionProbabilities {
    fn execute(
        &self,
        handles: &mut HandleContainer<
            <FusionCubeRuntime<CudaRuntime> as FusionRuntime>::FusionHandle,
        >,
    ) {
        type B = CubeBackend<CudaRuntime>;
        let ([scores, grad_probabilities, correction], [probabilities_ir, score_gradient_ir]) =
            self.desc.as_fixed();
        let (probabilities, score_gradient) = B::attention_backward_probabilities(
            handles.get_float_tensor::<B>(scores),
            handles.get_float_tensor::<B>(grad_probabilities),
            handles.get_float_tensor::<B>(correction),
            self.scale,
            self.row_offset,
            self.causal,
        );
        handles.register_float_tensor::<B>(&probabilities_ir.id, probabilities);
        handles.register_float_tensor::<B>(&score_gradient_ir.id, score_gradient);
    }
}

impl AttentionBackend for Fusion<CubeBackend<CudaRuntime>> {
    fn attention_inner(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        causal: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let client = query.client.clone();
        let shape = query.shape.clone();
        let dtype = query.dtype;
        let f16 = DType::F16;
        let outputs = [
            TensorIr::uninit(client.create_empty_handle(), shape.clone(), dtype),
            TensorIr::uninit(client.create_empty_handle(), shape.clone(), f16),
            TensorIr::uninit(client.create_empty_handle(), key.shape.clone(), f16),
            TensorIr::uninit(client.create_empty_handle(), value.shape.clone(), f16),
            TensorIr::uninit(client.create_empty_handle(), shape, f16),
        ];
        let desc = CustomOpIr::with_scalars(
            "hermes_flash_attention",
            &[query.into_ir(), key.into_ir(), value.into_ir()],
            &outputs,
            vec![ScalarIr::Bool(causal)],
        );
        let [output, query, key, value, saved_output] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                AttentionForward { desc, causal },
            )
            .try_into()
            .expect("attention forward has five outputs");
        (output, query, key, value, saved_output)
    }

    fn attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        causal: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        // Keep the recompute graph visible to Burn so its softmax correction,
        // casts, and reductions fuse around the accelerated matmuls.
        chunked_attention_backward::<Self>(
            query,
            key,
            value,
            output,
            grad_output,
            causal,
            ATTENTION_BACKWARD_CHUNK_ROWS,
        )
    }

    fn attention_backward_probabilities(
        scores: FloatTensor<Self>,
        grad_probabilities: FloatTensor<Self>,
        correction: FloatTensor<Self>,
        scale: f32,
        row_offset: usize,
        causal: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let client = scores.client.clone();
        let shape = scores.shape.clone();
        let probabilities =
            TensorIr::uninit(client.create_empty_handle(), shape.clone(), DType::BF16);
        let score_gradient = TensorIr::uninit(client.create_empty_handle(), shape, DType::BF16);
        let desc = CustomOpIr::with_scalars(
            "hermes_attention_backward_probabilities",
            &[
                scores.into_ir(),
                grad_probabilities.into_ir(),
                correction.into_ir(),
            ],
            &[probabilities, score_gradient],
            vec![
                ScalarIr::Float(scale as f64),
                ScalarIr::UInt(row_offset as u64),
                ScalarIr::Bool(causal),
            ],
        );
        let [probabilities, score_gradient] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                AttentionProbabilities {
                    desc,
                    scale,
                    row_offset,
                    causal,
                },
            )
            .try_into()
            .expect("attention probabilities have two outputs");
        (probabilities, score_gradient)
    }
    fn attention_causal_mask(scores: FloatTensor<Self>, row_offset: usize) -> FloatTensor<Self> {
        let client = scores.client.clone();
        let output = TensorIr::uninit(
            client.create_empty_handle(),
            scores.shape.clone(),
            scores.dtype,
        );
        let desc = CustomOpIr::with_scalars(
            "hermes_attention_causal_mask",
            &[scores.into_ir()],
            &[output],
            vec![ScalarIr::UInt(row_offset as u64)],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                AttentionCausalMask { desc, row_offset },
            )
            .remove(0)
    }
}
