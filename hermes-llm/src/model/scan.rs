//! Fused Mamba selective scan.
//!
//! CPU uses a tensor-op reference. GPU inference and training use CubeCL
//! forward and backward kernels directly on Burn's resident tensors.

use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{TensorPrimitive, ops::FloatTensor};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

use super::conv::DepthwiseConv1dBackend;

fn stable_softplus<B: Backend, const D: usize>(value: Tensor<B, D>) -> Tensor<B, D> {
    relu(value.clone()) + (-value.abs()).exp().add_scalar(1.0).log()
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ScanOutput<B: Backend> {
    pub(super) y: FloatTensor<B>,
    pub(super) h: FloatTensor<B>,
    pub(super) states: Option<FloatTensor<B>>,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ScanGradients<B: Backend> {
    pub(super) delta: FloatTensor<B>,
    pub(super) xs: FloatTensor<B>,
    pub(super) b_mat: FloatTensor<B>,
    pub(super) c_mat: FloatTensor<B>,
    pub(super) a: FloatTensor<B>,
    pub(super) d: FloatTensor<B>,
    pub(super) h: FloatTensor<B>,
}

/// Backend capability required by the Mamba mixer.
pub trait MambaBackend: Backend + DepthwiseConv1dBackend {
    #[allow(clippy::too_many_arguments)]
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
    ) -> ScanOutput<Self> {
        reference_selective_scan(delta, xs, b_mat, c_mat, a, d, h, state_dim, save_states)
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> ScanGradients<Self> {
        reference_selective_scan_backward(
            delta, xs, b_mat, c_mat, a, d, h, states, grad_y, state_dim,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn selective_scan<B: MambaBackend>(
    delta: Tensor<B, 3>,
    xs: Tensor<B, 3>,
    b_mat: Tensor<B, 3>,
    c_mat: Tensor<B, 3>,
    a: Tensor<B, 2>,
    d: Tensor<B, 1>,
    h: Tensor<B, 3>,
    state_dim: usize,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let output = B::selective_scan_inner(
        delta.into_primitive().tensor(),
        xs.into_primitive().tensor(),
        b_mat.into_primitive().tensor(),
        c_mat.into_primitive().tensor(),
        a.into_primitive().tensor(),
        d.into_primitive().tensor(),
        h.into_primitive().tensor(),
        state_dim,
        false,
    );
    (
        Tensor::from_primitive(TensorPrimitive::Float(output.y)),
        Tensor::from_primitive(TensorPrimitive::Float(output.h)),
    )
}

#[allow(clippy::too_many_arguments)]
fn reference_selective_scan<B: Backend>(
    delta: FloatTensor<B>,
    xs: FloatTensor<B>,
    b_mat: FloatTensor<B>,
    c_mat: FloatTensor<B>,
    a: FloatTensor<B>,
    d: FloatTensor<B>,
    h: FloatTensor<B>,
    state_dim: usize,
    save_states: bool,
) -> ScanOutput<B> {
    let delta = stable_softplus(Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(
        delta,
    )));
    let xs = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(xs));
    let b_mat = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(b_mat));
    let c_mat = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(c_mat));
    let a = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(a));
    let d = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(d));
    let mut h = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(h));
    let [batch, seq_len, channels] = xs.dims();
    assert_eq!(state_dim, h.dims()[2]);

    let mut ys = Vec::with_capacity(seq_len);
    let mut states = save_states.then(|| Vec::with_capacity(seq_len));
    for t in 0..seq_len {
        let dt = delta
            .clone()
            .slice([0..batch, t..t + 1, 0..channels])
            .reshape([batch, channels]);
        let xt = xs
            .clone()
            .slice([0..batch, t..t + 1, 0..channels])
            .reshape([batch, channels]);
        let bt = b_mat
            .clone()
            .slice([0..batch, t..t + 1, 0..state_dim])
            .reshape([batch, state_dim]);
        let ct = c_mat
            .clone()
            .slice([0..batch, t..t + 1, 0..state_dim])
            .reshape([batch, state_dim]);

        let dt_e = dt.unsqueeze_dim::<3>(2);
        let da = (dt_e.clone() * a.clone().unsqueeze_dim::<3>(0)).exp();
        let dbx = dt_e * bt.unsqueeze_dim::<3>(1) * xt.clone().unsqueeze_dim::<3>(2);
        h = h * da + dbx;
        let yt = (h.clone() * ct.unsqueeze_dim::<3>(1))
            .sum_dim(2)
            .reshape([batch, channels])
            + xt * d.clone().unsqueeze_dim::<2>(0);
        ys.push(yt.unsqueeze_dim::<3>(1));
        if let Some(states) = states.as_mut() {
            states.push(h.clone().unsqueeze_dim::<4>(1));
        }
    }

    ScanOutput {
        y: Tensor::cat(ys, 1).into_primitive().tensor(),
        h: h.into_primitive().tensor(),
        states: states.map(|states| Tensor::cat(states, 1).into_primitive().tensor()),
    }
}

#[allow(clippy::too_many_arguments)]
fn reference_selective_scan_backward<B: Backend>(
    delta: FloatTensor<B>,
    xs: FloatTensor<B>,
    b_mat: FloatTensor<B>,
    c_mat: FloatTensor<B>,
    a: FloatTensor<B>,
    d: FloatTensor<B>,
    h: FloatTensor<B>,
    states: FloatTensor<B>,
    grad_y: FloatTensor<B>,
    state_dim: usize,
) -> ScanGradients<B> {
    let delta_raw = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(delta));
    let delta_derivative = sigmoid(delta_raw.clone());
    let delta = stable_softplus(delta_raw);
    let xs = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(xs));
    let b_mat = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(b_mat));
    let c_mat = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(c_mat));
    let a = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(a));
    let d = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(d));
    let h0 = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(h));
    let states = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(states));
    let grad_y = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(grad_y));
    let [batch, seq_len, channels] = xs.dims();
    let device = xs.device();

    let mut grad_a = Tensor::zeros([channels, state_dim], &device);
    let mut grad_d = Tensor::zeros([channels], &device);
    let mut grad_h = Tensor::zeros([batch, channels, state_dim], &device);
    let mut grad_delta = Vec::with_capacity(seq_len);
    let mut grad_xs = Vec::with_capacity(seq_len);
    let mut grad_b = Vec::with_capacity(seq_len);
    let mut grad_c = Vec::with_capacity(seq_len);

    for step in 0..seq_len {
        let t = seq_len - step - 1;
        let dt = delta
            .clone()
            .slice([0..batch, t..t + 1, 0..channels])
            .reshape([batch, channels]);
        let xt = xs
            .clone()
            .slice([0..batch, t..t + 1, 0..channels])
            .reshape([batch, channels]);
        let bt = b_mat
            .clone()
            .slice([0..batch, t..t + 1, 0..state_dim])
            .reshape([batch, state_dim]);
        let ct = c_mat
            .clone()
            .slice([0..batch, t..t + 1, 0..state_dim])
            .reshape([batch, state_dim]);
        let dy = grad_y
            .clone()
            .slice([0..batch, t..t + 1, 0..channels])
            .reshape([batch, channels]);
        let h_t = states
            .clone()
            .slice([0..batch, t..t + 1, 0..channels, 0..state_dim])
            .reshape([batch, channels, state_dim]);
        let h_prev = if t == 0 {
            h0.clone()
        } else {
            states
                .clone()
                .slice([0..batch, t - 1..t, 0..channels, 0..state_dim])
                .reshape([batch, channels, state_dim])
        };

        let dy_e = dy.clone().unsqueeze_dim::<3>(2);
        let dt_e = dt.clone().unsqueeze_dim::<3>(2);
        let bt_e = bt.unsqueeze_dim::<3>(1);
        let alpha = (dt_e.clone() * a.clone().unsqueeze_dim::<3>(0)).exp();
        let grad_ct = (dy_e.clone() * h_t).sum_dim(1).reshape([batch, state_dim]);
        grad_h = grad_h + dy_e * ct.unsqueeze_dim::<3>(1);

        let grad_alpha = grad_h.clone() * h_prev;
        let grad_dt = (grad_alpha.clone() * alpha.clone() * a.clone().unsqueeze_dim::<3>(0)
            + grad_h.clone() * bt_e.clone() * xt.clone().unsqueeze_dim::<3>(2))
        .sum_dim(2)
        .reshape([batch, channels]);
        let grad_xt = dy.clone() * d.clone().unsqueeze_dim::<2>(0)
            + (grad_h.clone() * dt_e.clone() * bt_e.clone())
                .sum_dim(2)
                .reshape([batch, channels]);
        let grad_bt = (grad_h.clone() * dt_e.clone() * xt.unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch, state_dim]);
        grad_a = grad_a
            + (grad_alpha * alpha.clone() * dt_e)
                .sum_dim(0)
                .reshape([channels, state_dim]);
        grad_d = grad_d
            + (dy.clone()
                * xs.clone()
                    .slice([0..batch, t..t + 1, 0..channels])
                    .reshape([batch, channels]))
            .sum_dim(0)
            .reshape([channels]);
        grad_h = grad_h * alpha;

        grad_delta.push(grad_dt.unsqueeze_dim::<3>(1));
        grad_xs.push(grad_xt.unsqueeze_dim::<3>(1));
        grad_b.push(grad_bt.unsqueeze_dim::<3>(1));
        grad_c.push(grad_ct.unsqueeze_dim::<3>(1));
    }

    grad_delta.reverse();
    grad_xs.reverse();
    grad_b.reverse();
    grad_c.reverse();
    ScanGradients {
        delta: (Tensor::cat(grad_delta, 1) * delta_derivative)
            .into_primitive()
            .tensor(),
        xs: Tensor::cat(grad_xs, 1).into_primitive().tensor(),
        b_mat: Tensor::cat(grad_b, 1).into_primitive().tensor(),
        c_mat: Tensor::cat(grad_c, 1).into_primitive().tensor(),
        a: grad_a.into_primitive().tensor(),
        d: grad_d.into_primitive().tensor(),
        h: grad_h.into_primitive().tensor(),
    }
}

impl MambaBackend for burn_ndarray::NdArray {}

#[derive(Clone, Debug)]
struct SelectiveScanState<B: MambaBackend> {
    delta: FloatTensor<B>,
    xs: FloatTensor<B>,
    b_mat: FloatTensor<B>,
    c_mat: FloatTensor<B>,
    a: FloatTensor<B>,
    d: FloatTensor<B>,
    h: FloatTensor<B>,
    states: FloatTensor<B>,
    state_dim: usize,
}

#[derive(Debug)]
struct SelectiveScanBackward;

impl<B: MambaBackend> Backward<B, 7> for SelectiveScanBackward {
    type State = SelectiveScanState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 7>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_delta, node_xs, node_b, node_c, node_a, node_d, node_h] = ops.parents;
        let grad_y = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let output = B::selective_scan_backward(
            state.delta,
            state.xs,
            state.b_mat,
            state.c_mat,
            state.a,
            state.d,
            state.h,
            state.states,
            grad_y,
            state.state_dim,
        );

        for (node, grad) in [
            (node_delta, output.delta),
            (node_xs, output.xs),
            (node_b, output.b_mat),
            (node_c, output.c_mat),
            (node_a, output.a),
            (node_d, output.d),
            (node_h, output.h),
        ] {
            if let Some(node) = node {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

impl<B: MambaBackend, C: CheckpointStrategy> MambaBackend for Autodiff<B, C> {
    fn selective_scan_inner(
        delta: FloatTensor<Self>,
        xs: FloatTensor<Self>,
        b_mat: FloatTensor<Self>,
        c_mat: FloatTensor<Self>,
        a: FloatTensor<Self>,
        d: FloatTensor<Self>,
        h: FloatTensor<Self>,
        state_dim: usize,
        _save_states: bool,
    ) -> ScanOutput<Self> {
        match SelectiveScanBackward
            .prepare::<C>([
                delta.node.clone(),
                xs.node.clone(),
                b_mat.node.clone(),
                c_mat.node.clone(),
                a.node.clone(),
                d.node.clone(),
                h.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output = B::selective_scan_inner(
                    delta.primitive.clone(),
                    xs.primitive.clone(),
                    b_mat.primitive.clone(),
                    c_mat.primitive.clone(),
                    a.primitive.clone(),
                    d.primitive.clone(),
                    h.primitive.clone(),
                    state_dim,
                    true,
                );
                let state = SelectiveScanState {
                    delta: delta.primitive,
                    xs: xs.primitive,
                    b_mat: b_mat.primitive,
                    c_mat: c_mat.primitive,
                    a: a.primitive,
                    d: d.primitive,
                    h: h.primitive,
                    states: output
                        .states
                        .expect("training selective scan must retain recurrent states"),
                    state_dim,
                };
                ScanOutput {
                    y: prep.finish(state, output.y),
                    h: <Self as burn::tensor::backend::AutodiffBackend>::from_inner(output.h),
                    states: None,
                }
            }
            OpsKind::UnTracked(prep) => {
                let output = B::selective_scan_inner(
                    delta.primitive,
                    xs.primitive,
                    b_mat.primitive,
                    c_mat.primitive,
                    a.primitive,
                    d.primitive,
                    h.primitive,
                    state_dim,
                    false,
                );
                ScanOutput {
                    y: prep.finish(output.y),
                    h: <Self as burn::tensor::backend::AutodiffBackend>::from_inner(output.h),
                    states: None,
                }
            }
        }
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use burn::tensor::{Shape, TensorMetadata};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::element::{BoolElement, FloatElement, IntElement};
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};

    use super::{MambaBackend, ScanGradients, ScanOutput};
    use crate::model::cube_tensor::{empty_like, into_contiguous, zeros_like};

    const THREADS_PER_CUBE: u32 = 128;
    const PLANE_WIDTH: u32 = 32;
    const PLANES_PER_CUBE: u32 = THREADS_PER_CUBE / PLANE_WIDTH;

    #[cube]
    fn atomic_add_f32(target: &mut Atomic<f32>, value: f32) {
        target.fetch_add(value);
    }

    #[cube]
    fn stable_softplus(value: f32) -> f32 {
        if value > 0.0 {
            value + ((-value).exp() + 1.0).ln()
        } else {
            (value.exp() + 1.0).ln()
        }
    }

    #[cube]
    fn stable_sigmoid(value: f32) -> f32 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp_value = value.exp();
            exp_value / (1.0 + exp_value)
        }
    }

    #[cube(launch)]
    fn softplus_forward(input: &Array<f32>, output: &mut Array<f32>) {
        let idx = ABSOLUTE_POS;
        if idx < input.len() {
            output[idx] = stable_softplus(input[idx]);
        }
    }

    #[cube(launch)]
    fn selective_scan_step(
        delta: &Array<f32>,
        xs: &Array<f32>,
        b_mat: &Array<f32>,
        c_mat: &Array<f32>,
        a: &Array<f32>,
        d: &Array<f32>,
        h_in: &Array<f32>,
        y: &mut Array<f32>,
        h_out: &mut Array<f32>,
        channels: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let idx = ABSOLUTE_POS;
        let total = xs.len();
        if idx < total {
            let batch = idx / channels;
            let channel = idx % channels;
            let state_base = idx * state_dim;
            let a_base = channel * state_dim;
            let mut state = Array::<f32>::new(state_dim);
            for n in 0..state_dim {
                state[n] = h_in[state_base + n];
            }

            let btn = batch * state_dim;
            let dt = delta[idx];
            let x = xs[idx];
            let mut out = 0.0f32;
            for n in 0..state_dim {
                let da = (dt * a[a_base + n]).exp();
                state[n] = state[n] * da + dt * b_mat[btn + n] * x;
                out += state[n] * c_mat[btn + n];
                h_out[state_base + n] = state[n];
            }
            y[idx] = out + x * d[channel];
        }
    }

    /// Parallelize the recurrent state dimension. Each thread still follows
    /// one exact scalar recurrence through time, while a separate kernel
    /// reduces the small state dimension into the output.
    #[cube(launch)]
    fn selective_scan_states(
        delta: &Array<f32>,
        xs: &Array<f32>,
        b_mat: &Array<f32>,
        a: &Array<f32>,
        h_in: &Array<f32>,
        states: &mut Array<f32>,
        h_out: &mut Array<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let idx = ABSOLUTE_POS;
        if idx < h_out.len() {
            let n = idx % state_dim;
            let batch_channel = idx / state_dim;
            let batch = batch_channel / channels;
            let channel = batch_channel % channels;
            let mut state = h_in[idx];
            let av = a[channel * state_dim + n];

            for t in 0..seq_len {
                let btc = (batch * seq_len + t) * channels + channel;
                let btn = (batch * seq_len + t) * state_dim + n;
                let dt = delta[btc];
                let x = xs[btc];
                state = state * (dt * av).exp() + dt * b_mat[btn] * x;
                states[btc * state_dim + n] = state;
            }
            h_out[idx] = state;
        }
    }

    #[cube(launch)]
    fn selective_scan_output(
        xs: &Array<f32>,
        c_mat: &Array<f32>,
        d: &Array<f32>,
        states: &Array<f32>,
        y: &mut Array<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let btc = ABSOLUTE_POS;
        if btc < xs.len() {
            let channel = btc % channels;
            let batch_time = btc / channels;
            let batch = batch_time / seq_len;
            let t = batch_time % seq_len;
            let btn = (batch * seq_len + t) * state_dim;
            let mut out = 0.0f32;
            for n in 0..state_dim {
                out += states[btc * state_dim + n] * c_mat[btn + n];
            }
            y[btc] = out + xs[btc] * d[channel];
        }
    }

    #[allow(clippy::useless_conversion)]
    #[cube(launch)]
    fn selective_scan_bc_backward(
        delta: &Array<f32>,
        xs: &Array<f32>,
        c_mat: &Array<f32>,
        a: &Array<f32>,
        h_in: &Array<f32>,
        states: &Array<f32>,
        grad_y: &Array<f32>,
        grad_b: &mut Array<Atomic<f32>>,
        grad_c: &mut Array<Atomic<f32>>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let state_group = ABSOLUTE_POS_Y as usize;
        let channel = ABSOLUTE_POS_X as usize;
        let total_state_groups = h_in.len() / channels;
        let active = channel < channels && state_group < total_state_groups;
        let n = state_group % state_dim;
        let batch = state_group / state_dim;
        let a_index = channel * state_dim + n;
        let mut adjoint = 0.0f32;

        for step in 0..seq_len {
            let t = seq_len - step - 1;
            let btc = (batch * seq_len + t) * channels + channel;
            let btn = (batch * seq_len + t) * state_dim;
            let dt = if active { delta[btc] } else { 0.0f32.into() };
            let x = if active { xs[btc] } else { 0.0f32.into() };
            let dy = if active { grad_y[btc] } else { 0.0f32.into() };
            let av = if active { a[a_index] } else { 0.0f32.into() };
            let cv = if active {
                c_mat[btn + n]
            } else {
                0.0f32.into()
            };
            let alpha = (dt * av).exp();
            let h_t = if active {
                states[btc * state_dim + n]
            } else {
                0.0f32.into()
            };
            let g = adjoint + dy * cv;
            let grad_b_sum = plane_sum(g * dt * x);
            let grad_c_sum = plane_sum(dy * h_t);
            if UNIT_POS_PLANE == 0 && state_group < total_state_groups {
                atomic_add_f32(&mut grad_b[btn + n], grad_b_sum);
                atomic_add_f32(&mut grad_c[btn + n], grad_c_sum);
            }
            adjoint = g * alpha;
        }
    }

    /// A plane owns one (batch, channel) recurrence and reduces its state
    /// lanes without global atomics for the per-token input gradients.
    #[allow(clippy::useless_conversion)]
    #[cube(launch)]
    fn selective_scan_input_backward(
        delta: &Array<f32>,
        delta_raw: &Array<f32>,
        xs: &Array<f32>,
        b_mat: &Array<f32>,
        c_mat: &Array<f32>,
        a: &Array<f32>,
        d: &Array<f32>,
        h_in: &Array<f32>,
        states: &Array<f32>,
        grad_y: &Array<f32>,
        grad_delta: &mut Array<f32>,
        grad_xs: &mut Array<f32>,
        grad_a: &mut Array<Atomic<f32>>,
        grad_d: &mut Array<Atomic<f32>>,
        grad_h: &mut Array<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let batch_channel = ABSOLUTE_POS_Y as usize;
        let n = UNIT_POS_PLANE as usize;
        let total_batch_channels = h_in.len() / state_dim;
        let active_channel = batch_channel < total_batch_channels;
        let active = active_channel && n < state_dim;
        let batch = batch_channel / channels;
        let channel = batch_channel % channels;
        let state_index = batch_channel * state_dim + n;
        let a_index = channel * state_dim + n;
        let mut adjoint = 0.0f32;
        let mut grad_a_local = 0.0f32;
        let mut grad_d_local = 0.0f32;

        for step in 0..seq_len {
            let t = seq_len - step - 1;
            let btc = (batch * seq_len + t) * channels + channel;
            let btn = (batch * seq_len + t) * state_dim;
            let dt = if active { delta[btc] } else { 0.0f32.into() };
            let raw_dt = if active {
                delta_raw[btc]
            } else {
                0.0f32.into()
            };
            let x = if active { xs[btc] } else { 0.0f32.into() };
            let dy = if active { grad_y[btc] } else { 0.0f32.into() };
            let av = if active { a[a_index] } else { 0.0f32.into() };
            let bv = if active {
                b_mat[btn + n]
            } else {
                0.0f32.into()
            };
            let cv = if active {
                c_mat[btn + n]
            } else {
                0.0f32.into()
            };
            let alpha = (dt * av).exp();
            let h_prev = if active {
                if t == 0 {
                    h_in[state_index]
                } else {
                    states[((batch * seq_len + t - 1) * channels + channel) * state_dim + n]
                }
            } else {
                0.0f32.into()
            };
            let g = adjoint + dy * cv;
            let grad_dt = plane_sum(g * (h_prev * alpha * av + bv * x));
            let grad_x = plane_sum(g * dt * bv);
            if UNIT_POS_PLANE == 0 && active_channel {
                grad_delta[btc] = grad_dt * stable_sigmoid(raw_dt);
                grad_xs[btc] = dy * d[channel] + grad_x;
                grad_d_local += dy * x;
            }
            grad_a_local += g * h_prev * alpha * dt;
            adjoint = g * alpha;
        }

        if active {
            atomic_add_f32(&mut grad_a[a_index], grad_a_local);
            grad_h[state_index] = adjoint;
        }
        if UNIT_POS_PLANE == 0 && active_channel {
            atomic_add_f32(&mut grad_d[channel], grad_d_local);
        }
    }

    impl<R, F, I, BT> MambaBackend for CubeBackend<R, F, I, BT>
    where
        R: CubeRuntime,
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        fn selective_scan_inner(
            delta: CubeTensor<R>,
            xs: CubeTensor<R>,
            b_mat: CubeTensor<R>,
            c_mat: CubeTensor<R>,
            a: CubeTensor<R>,
            d: CubeTensor<R>,
            h: CubeTensor<R>,
            state_dim: usize,
            save_states: bool,
        ) -> ScanOutput<Self> {
            let [batch, seq_len, channels] = xs.shape().dims();
            let delta_raw = into_contiguous(delta);
            let xs = into_contiguous(xs);
            let b_mat = into_contiguous(b_mat);
            let c_mat = into_contiguous(c_mat);
            let a = into_contiguous(a);
            let d = into_contiguous(d);
            let h = into_contiguous(h);
            let y = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let h_out = empty_like(&xs, Shape::new([batch, channels, state_dim]));
            let delta = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let parallel_scan = save_states || seq_len > 1;
            let states = if parallel_scan {
                empty_like(&xs, Shape::new([batch, seq_len, channels, state_dim]))
            } else {
                h_out.clone()
            };

            let client = xs.client.clone();
            let delta_total = (batch * seq_len * channels) as u32;
            softplus_forward::launch::<R>(
                &client,
                CubeCount::Static(delta_total.div_ceil(THREADS_PER_CUBE), 1, 1),
                CubeDim::new_1d(THREADS_PER_CUBE),
                delta_raw.into_array_arg(),
                delta.clone().into_array_arg(),
            );
            if parallel_scan {
                let state_total = (batch * channels * state_dim) as u32;
                selective_scan_states::launch::<R>(
                    &client,
                    CubeCount::Static(state_total.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.clone().into_array_arg(),
                    xs.clone().into_array_arg(),
                    b_mat.clone().into_array_arg(),
                    a.clone().into_array_arg(),
                    h.clone().into_array_arg(),
                    states.clone().into_array_arg(),
                    h_out.clone().into_array_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                );
                let output_total = (batch * seq_len * channels) as u32;
                selective_scan_output::launch::<R>(
                    &client,
                    CubeCount::Static(output_total.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    xs.clone().into_array_arg(),
                    c_mat.into_array_arg(),
                    d.into_array_arg(),
                    states.clone().into_array_arg(),
                    y.clone().into_array_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                );
            } else {
                let total = (batch * channels) as u32;
                selective_scan_step::launch::<R>(
                    &client,
                    CubeCount::Static(total.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.into_array_arg(),
                    xs.clone().into_array_arg(),
                    b_mat.into_array_arg(),
                    c_mat.into_array_arg(),
                    a.into_array_arg(),
                    d.into_array_arg(),
                    h.into_array_arg(),
                    y.clone().into_array_arg(),
                    h_out.clone().into_array_arg(),
                    channels as u32,
                    state_dim,
                );
            }

            ScanOutput {
                y,
                h: h_out,
                states: save_states.then_some(states),
            }
        }

        fn selective_scan_backward(
            delta: CubeTensor<R>,
            xs: CubeTensor<R>,
            b_mat: CubeTensor<R>,
            c_mat: CubeTensor<R>,
            a: CubeTensor<R>,
            d: CubeTensor<R>,
            h: CubeTensor<R>,
            states: CubeTensor<R>,
            grad_y: CubeTensor<R>,
            state_dim: usize,
        ) -> ScanGradients<Self> {
            let [batch, seq_len, channels] = xs.shape().dims();
            assert!(
                state_dim <= PLANE_WIDTH as usize,
                "GPU selective scan supports at most {PLANE_WIDTH} states"
            );
            let delta_raw = into_contiguous(delta);
            let xs = into_contiguous(xs);
            let b_mat = into_contiguous(b_mat);
            let c_mat = into_contiguous(c_mat);
            let a = into_contiguous(a);
            let d = into_contiguous(d);
            let h = into_contiguous(h);
            let states = into_contiguous(states);
            let grad_y = into_contiguous(grad_y);
            let grad_delta = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let grad_xs = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let grad_b = zeros_like(&xs, Shape::new([batch, seq_len, state_dim]));
            let grad_c = zeros_like(&xs, Shape::new([batch, seq_len, state_dim]));
            let grad_a = zeros_like(&xs, Shape::new([channels, state_dim]));
            let grad_d = zeros_like(&xs, Shape::new([channels]));
            let grad_h = empty_like(&xs, Shape::new([batch, channels, state_dim]));
            let delta = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let client = xs.client.clone();

            let delta_total = (batch * seq_len * channels) as u32;
            softplus_forward::launch::<R>(
                &client,
                CubeCount::Static(delta_total.div_ceil(THREADS_PER_CUBE), 1, 1),
                CubeDim::new_1d(THREADS_PER_CUBE),
                delta_raw.clone().into_array_arg(),
                delta.clone().into_array_arg(),
            );

            selective_scan_bc_backward::launch::<R>(
                &client,
                CubeCount::Static(
                    (channels as u32).div_ceil(PLANE_WIDTH),
                    (batch as u32 * state_dim as u32).div_ceil(PLANES_PER_CUBE),
                    1,
                ),
                CubeDim::new_2d(PLANE_WIDTH, PLANES_PER_CUBE),
                delta.clone().into_array_arg(),
                xs.clone().into_array_arg(),
                c_mat.clone().into_array_arg(),
                a.clone().into_array_arg(),
                h.clone().into_array_arg(),
                states.clone().into_array_arg(),
                grad_y.clone().into_array_arg(),
                grad_b.clone().into_array_arg(),
                grad_c.clone().into_array_arg(),
                channels as u32,
                seq_len as u32,
                state_dim,
            );
            selective_scan_input_backward::launch::<R>(
                &client,
                CubeCount::Static(
                    1,
                    (batch as u32 * channels as u32).div_ceil(PLANES_PER_CUBE),
                    1,
                ),
                CubeDim::new_2d(PLANE_WIDTH, PLANES_PER_CUBE),
                delta.into_array_arg(),
                delta_raw.into_array_arg(),
                xs.into_array_arg(),
                b_mat.into_array_arg(),
                c_mat.into_array_arg(),
                a.into_array_arg(),
                d.into_array_arg(),
                h.into_array_arg(),
                states.into_array_arg(),
                grad_y.into_array_arg(),
                grad_delta.clone().into_array_arg(),
                grad_xs.clone().into_array_arg(),
                grad_a.clone().into_array_arg(),
                grad_d.clone().into_array_arg(),
                grad_h.clone().into_array_arg(),
                channels as u32,
                seq_len as u32,
                state_dim,
            );

            ScanGradients {
                delta: grad_delta,
                xs: grad_xs,
                b_mat: grad_b,
                c_mat: grad_c,
                a: grad_a,
                d: grad_d,
                h: grad_h,
            }
        }
    }
}

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod tests {
    use burn::tensor::{Tensor, TensorData};
    use burn_autodiff::Autodiff;
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    use burn_cuda::Cuda as Gpu;
    use burn_ndarray::NdArray;
    #[cfg(feature = "metal")]
    use burn_wgpu::Wgpu;

    use super::selective_scan;
    use crate::model::test_support::{max_diff, values};

    #[test]
    fn test_cubecl_selective_scan_matches_ndarray_reference() {
        type Cpu = NdArray;
        #[cfg(feature = "metal")]
        type GpuBackend = Wgpu;
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        type GpuBackend = Gpu<cubecl::flex32>;
        let cpu = Default::default();
        let gpu = Default::default();
        let (batch, seq_len, channels, state_dim) = (2, 5, 3, 4);

        let delta = values(batch * seq_len * channels, 0.13, 0.08);
        let xs = values(batch * seq_len * channels, 0.17, -0.03);
        let b_mat = values(batch * seq_len * state_dim, 0.19, 0.02);
        let c_mat = values(batch * seq_len * state_dim, 0.23, -0.01);
        let a = values(channels * state_dim, 0.11, -0.4);
        let d = values(channels, 0.07, 0.9);
        let h = values(batch * channels * state_dim, 0.05, 0.01);

        macro_rules! tensor {
            ($backend:ty, $device:expr, $data:expr, $shape:expr) => {
                Tensor::<$backend, _>::from_data(TensorData::new($data.clone(), $shape), $device)
            };
        }

        let (cpu_y, cpu_h) = selective_scan(
            tensor!(Cpu, &cpu, delta, [batch, seq_len, channels]),
            tensor!(Cpu, &cpu, xs, [batch, seq_len, channels]),
            tensor!(Cpu, &cpu, b_mat, [batch, seq_len, state_dim]),
            tensor!(Cpu, &cpu, c_mat, [batch, seq_len, state_dim]),
            tensor!(Cpu, &cpu, a, [channels, state_dim]),
            tensor!(Cpu, &cpu, d, [channels]),
            tensor!(Cpu, &cpu, h, [batch, channels, state_dim]),
            state_dim,
        );
        let (gpu_y, gpu_h) = selective_scan(
            tensor!(GpuBackend, &gpu, delta, [batch, seq_len, channels]),
            tensor!(GpuBackend, &gpu, xs, [batch, seq_len, channels]),
            tensor!(GpuBackend, &gpu, b_mat, [batch, seq_len, state_dim]),
            tensor!(GpuBackend, &gpu, c_mat, [batch, seq_len, state_dim]),
            tensor!(GpuBackend, &gpu, a, [channels, state_dim]),
            tensor!(GpuBackend, &gpu, d, [channels]),
            tensor!(GpuBackend, &gpu, h, [batch, channels, state_dim]),
            state_dim,
        );

        assert!(max_diff(cpu_y.into_data(), gpu_y.into_data()) < 1e-5);
        assert!(max_diff(cpu_h.into_data(), gpu_h.into_data()) < 1e-5);
    }

    #[test]
    fn test_cubecl_selective_scan_backward_matches_ndarray() {
        type Cpu = Autodiff<NdArray>;
        #[cfg(feature = "metal")]
        type GpuBackend = Autodiff<Wgpu>;
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        type GpuBackend = Autodiff<Gpu<cubecl::flex32>>;
        let cpu = Default::default();
        let gpu = Default::default();
        let (batch, seq_len, channels, state_dim) = (2, 4, 3, 4);
        let shapes = (
            [batch, seq_len, channels],
            [batch, seq_len, state_dim],
            [channels, state_dim],
            [channels],
            [batch, channels, state_dim],
        );
        let data = (
            values(batch * seq_len * channels, 0.13, 0.08),
            values(batch * seq_len * channels, 0.17, -0.03),
            values(batch * seq_len * state_dim, 0.19, 0.02),
            values(batch * seq_len * state_dim, 0.23, -0.01),
            values(channels * state_dim, 0.11, -0.4),
            values(channels, 0.07, 0.9),
            values(batch * channels * state_dim, 0.05, 0.01),
        );

        macro_rules! run {
            ($backend:ty, $device:expr) => {{
                let delta = Tensor::<$backend, 3>::from_data(
                    TensorData::new(data.0.clone(), shapes.0),
                    $device,
                )
                .require_grad();
                let xs = Tensor::<$backend, 3>::from_data(
                    TensorData::new(data.1.clone(), shapes.0),
                    $device,
                )
                .require_grad();
                let b = Tensor::<$backend, 3>::from_data(
                    TensorData::new(data.2.clone(), shapes.1),
                    $device,
                )
                .require_grad();
                let c = Tensor::<$backend, 3>::from_data(
                    TensorData::new(data.3.clone(), shapes.1),
                    $device,
                )
                .require_grad();
                let a = Tensor::<$backend, 2>::from_data(
                    TensorData::new(data.4.clone(), shapes.2),
                    $device,
                )
                .require_grad();
                let d = Tensor::<$backend, 1>::from_data(
                    TensorData::new(data.5.clone(), shapes.3),
                    $device,
                )
                .require_grad();
                let h = Tensor::<$backend, 3>::from_data(
                    TensorData::new(data.6.clone(), shapes.4),
                    $device,
                )
                .require_grad();
                let (y, _) = selective_scan(
                    delta.clone(),
                    xs.clone(),
                    b.clone(),
                    c.clone(),
                    a.clone(),
                    d.clone(),
                    h.clone(),
                    state_dim,
                );
                let weights = Tensor::<$backend, 3>::from_data(
                    TensorData::new(values(batch * seq_len * channels, 0.29, 0.5), shapes.0),
                    $device,
                );
                let mut grads = (y * weights).sum().backward();
                [
                    delta.grad_remove(&mut grads).unwrap().into_data(),
                    xs.grad_remove(&mut grads).unwrap().into_data(),
                    b.grad_remove(&mut grads).unwrap().into_data(),
                    c.grad_remove(&mut grads).unwrap().into_data(),
                    a.grad_remove(&mut grads).unwrap().into_data(),
                    d.grad_remove(&mut grads).unwrap().into_data(),
                    h.grad_remove(&mut grads).unwrap().into_data(),
                ]
            }};
        }

        let cpu_grads = run!(Cpu, &cpu);
        let gpu_grads = run!(GpuBackend, &gpu);
        for ((name, cpu), gpu) in ["delta", "xs", "b", "c", "a", "d", "h"]
            .into_iter()
            .zip(cpu_grads)
            .zip(gpu_grads)
        {
            let difference = max_diff(cpu, gpu);
            assert!(difference < 2e-4, "{name} gradient max diff: {difference}");
        }
    }
}
