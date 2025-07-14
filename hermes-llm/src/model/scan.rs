//! Fused Mamba selective scan.
//!
//! CPU uses a tensor-op reference. GPU inference and training use CubeCL
//! forward and backward kernels directly on Burn's resident tensors.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use burn::backend::NdArray;
use burn::backend::{
    Backend, Dispatch, DispatchKindConversion, DispatchTensor, backend_extension,
    tensor::FloatTensor,
};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

use super::conv::DepthwiseConv1dBackend;

#[cfg(any(feature = "metal", feature = "cuda", test))]
const CHECKPOINTED_SCAN_INTERVAL: usize = 32;
#[cfg(any(feature = "metal", feature = "cuda", test))]
const MAX_FULL_STATE_BATCH: usize = 2;
#[cfg(any(feature = "metal", feature = "cuda", test))]
const MAX_FULL_STATE_BYTES: usize = 512 * 1024 * 1024;

/// Keep every recurrent state when a small training batch fits a bounded
/// per-layer budget; larger batches trade one recurrence recompute for memory.
#[cfg(any(feature = "metal", feature = "cuda", test))]
pub(super) fn scan_checkpoint_interval(
    batch: usize,
    seq_len: usize,
    channels: usize,
    state_dim: usize,
) -> usize {
    let state_bytes = batch
        .saturating_mul(seq_len)
        .saturating_mul(channels)
        .saturating_mul(state_dim)
        .saturating_mul(size_of::<f32>());
    if batch <= MAX_FULL_STATE_BATCH && state_bytes <= MAX_FULL_STATE_BYTES {
        1
    } else {
        CHECKPOINTED_SCAN_INTERVAL
    }
}

fn stable_softplus<const D: usize>(value: Tensor<D>) -> Tensor<D> {
    relu(value.clone()) + (-value.abs()).exp().add_scalar(1.0).log()
}

/// Backend capability required by the Mamba mixer.
#[cfg_attr(feature = "cuda", backend_extension(Cuda, Autodiff))]
#[cfg_attr(
    all(not(feature = "cuda"), feature = "metal"),
    backend_extension(Metal, Autodiff)
)]
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal")),
    backend_extension(NdArray, Autodiff)
)]
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
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[allow(clippy::too_many_arguments, clippy::type_complexity, unused_variables)]
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
        panic!("selective scan only supports first-order autodiff")
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn selective_scan(
    delta: Tensor<3>,
    xs: Tensor<3>,
    b_mat: Tensor<3>,
    c_mat: Tensor<3>,
    a: Tensor<2>,
    d: Tensor<1>,
    h: Tensor<3>,
    state_dim: usize,
) -> (Tensor<3>, Tensor<3>) {
    if delta.device() == Device::ndarray() {
        let (y, h, _) =
            reference_selective_scan_tensor(delta, xs, b_mat, c_mat, a, d, h, state_dim, false);
        return (y, h);
    }
    let output = Dispatch::selective_scan_inner(
        delta.into_dispatch(),
        xs.into_dispatch(),
        b_mat.into_dispatch(),
        c_mat.into_dispatch(),
        a.into_dispatch(),
        d.into_dispatch(),
        h.into_dispatch(),
        state_dim,
        false,
    );
    (
        Tensor::from_dispatch(output.0),
        Tensor::from_dispatch(output.1),
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
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let (y, h, states) = reference_selective_scan_tensor(
        Tensor::<3>::from_primitive::<B>(delta),
        Tensor::<3>::from_primitive::<B>(xs),
        Tensor::<3>::from_primitive::<B>(b_mat),
        Tensor::<3>::from_primitive::<B>(c_mat),
        Tensor::<2>::from_primitive::<B>(a),
        Tensor::<1>::from_primitive::<B>(d),
        Tensor::<3>::from_primitive::<B>(h),
        state_dim,
        save_states,
    );
    (
        y.try_into_primitive::<B>()
            .expect("scan output stayed on its input backend"),
        h.try_into_primitive::<B>()
            .expect("scan state stayed on its input backend"),
        states
            .try_into_primitive::<B>()
            .expect("saved scan states stayed on their input backend"),
    )
}

#[allow(clippy::too_many_arguments)]
fn reference_selective_scan_tensor(
    delta: Tensor<3>,
    xs: Tensor<3>,
    b_mat: Tensor<3>,
    c_mat: Tensor<3>,
    a: Tensor<2>,
    d: Tensor<1>,
    mut h: Tensor<3>,
    state_dim: usize,
    save_states: bool,
) -> (Tensor<3>, Tensor<3>, Tensor<4>) {
    let delta = stable_softplus(delta);
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

    let states = states
        .map(|states| Tensor::cat(states, 1))
        .unwrap_or_else(|| h.clone().unsqueeze_dim::<4>(1));
    (Tensor::cat(ys, 1), h, states)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
) -> (
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let delta_raw = Tensor::<3>::from_primitive::<B>(delta);
    let delta_derivative = sigmoid(delta_raw.clone());
    let delta = stable_softplus(delta_raw);
    let xs = Tensor::<3>::from_primitive::<B>(xs);
    let b_mat = Tensor::<3>::from_primitive::<B>(b_mat);
    let c_mat = Tensor::<3>::from_primitive::<B>(c_mat);
    let a = Tensor::<2>::from_primitive::<B>(a);
    let d = Tensor::<1>::from_primitive::<B>(d);
    let h0 = Tensor::<3>::from_primitive::<B>(h);
    let states = Tensor::<4>::from_primitive::<B>(states);
    let grad_y = Tensor::<3>::from_primitive::<B>(grad_y);
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
    (
        (Tensor::cat(grad_delta, 1) * delta_derivative)
            .try_into_primitive::<B>()
            .expect("delta gradient stayed on its input backend"),
        Tensor::cat(grad_xs, 1)
            .try_into_primitive::<B>()
            .expect("input gradient stayed on its input backend"),
        Tensor::cat(grad_b, 1)
            .try_into_primitive::<B>()
            .expect("B gradient stayed on its input backend"),
        Tensor::cat(grad_c, 1)
            .try_into_primitive::<B>()
            .expect("C gradient stayed on its input backend"),
        grad_a
            .try_into_primitive::<B>()
            .expect("A gradient stayed on its input backend"),
        grad_d
            .try_into_primitive::<B>()
            .expect("D gradient stayed on its input backend"),
        grad_h
            .try_into_primitive::<B>()
            .expect("state gradient stayed on its input backend"),
    )
}

impl MambaBackend for burn_ndarray::NdArray {
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
        reference_selective_scan::<Self>(delta, xs, b_mat, c_mat, a, d, h, state_dim, save_states)
    }

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
        reference_selective_scan_backward::<Self>(
            delta, xs, b_mat, c_mat, a, d, h, states, grad_y, state_dim,
        )
    }
}

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
            (node_delta, output.0),
            (node_xs, output.1),
            (node_b, output.2),
            (node_c, output.3),
            (node_a, output.4),
            (node_d, output.5),
            (node_h, output.6),
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
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
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
                let (y, h_out, states) = B::selective_scan_inner(
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
                    states,
                    state_dim,
                };
                (
                    prep.finish(state, y),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out.clone()),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (y, h_out, _states) = B::selective_scan_inner(
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
                (
                    prep.finish(y),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out.clone()),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out),
                )
            }
        }
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use burn::backend::TensorMetadata;
    use burn::tensor::Shape;
    use burn_cubecl::cubecl::ir::FastMath;
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};

    use super::{MambaBackend, scan_checkpoint_interval};
    use crate::model::cube_tensor::{empty_like, into_contiguous, zeros_like};

    const THREADS_PER_CUBE: u32 = 128;
    const PLANE_WIDTH: u32 = 32;
    const FORWARD_CHANNELS: u32 = THREADS_PER_CUBE / PLANE_WIDTH;
    // A100 measurements favor serial recurrence once this many independent
    // blocks are available; smaller grids benefit from parallel state lanes.
    const SERIAL_SCAN_MIN_BLOCKS: u32 = 128;
    const BACKWARD_CHANNELS: usize = 16;

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

    #[cube]
    fn half_plane_sum(value: f32, lane: usize) -> f32 {
        let mut sum = value;
        let other = plane_shuffle_down(sum, 8);
        if lane < 8 {
            sum += other;
        }
        let other = plane_shuffle_down(sum, 4);
        if lane < 4 {
            sum += other;
        }
        let other = plane_shuffle_down(sum, 2);
        if lane < 2 {
            sum += other;
        }
        let other = plane_shuffle_down(sum, 1);
        if lane == 0 {
            sum += other;
        }
        sum
    }

    #[cube(launch)]
    fn softplus_forward(input: &Tensor<f32>, output: &mut Tensor<f32>) {
        let idx = ABSOLUTE_POS;
        if idx < input.len() {
            output[idx] = stable_softplus(input[idx]);
        }
    }

    #[cube(launch)]
    fn selective_scan_step(
        delta: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        d: &Tensor<f32>,
        h_in: &Tensor<f32>,
        y: &mut Tensor<f32>,
        h_out: &mut Tensor<f32>,
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

    /// One thread owns a batch/channel pair. This avoids a plane reduction
    /// when the batch/channel grid already has enough blocks to fill the GPU.
    #[allow(clippy::manual_div_ceil)]
    #[cube(launch)]
    fn selective_scan_forward_serial(
        delta: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        d: &Tensor<f32>,
        h_in: &Tensor<f32>,
        y: &mut Tensor<f32>,
        checkpoints: &mut Tensor<f32>,
        h_out: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] checkpoint_interval: usize,
        #[comptime] save_checkpoints: bool,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let idx = ABSOLUTE_POS;
        let batch_channels = xs.len() / seq_len;
        if idx < batch_channels {
            let batch = idx / channels;
            let channel = idx % channels;
            let state_base = idx * state_dim;
            let a_base = channel * state_dim;
            let checkpoint_count = (seq_len + checkpoint_interval - 1) / checkpoint_interval;
            let mut state = Array::<f32>::new(state_dim);
            for n in 0..state_dim {
                state[n] = h_in[state_base + n];
            }

            for t in 0..seq_len {
                let btc = (batch * seq_len + t) * channels + channel;
                let btn = (batch * seq_len + t) * state_dim;
                let dt = delta[btc];
                let x = xs[btc];
                let mut out = 0.0f32;
                for n in 0..state_dim {
                    state[n] = state[n] * (dt * a[a_base + n]).exp() + dt * b_mat[btn + n] * x;
                    out += state[n] * c_mat[btn + n];
                }
                y[btc] = out + x * d[channel];

                if save_checkpoints && ((t + 1) % checkpoint_interval == 0 || t + 1 == seq_len) {
                    let checkpoint = t / checkpoint_interval;
                    let checkpoint_base =
                        ((batch * checkpoint_count + checkpoint) * channels + channel) * state_dim;
                    for n in 0..state_dim {
                        checkpoints[checkpoint_base + n] = state[n];
                    }
                }
            }
            for n in 0..state_dim {
                h_out[state_base + n] = state[n];
            }
        }
    }

    /// One plane owns a batch/channel pair. This exposes the state dimension
    /// as parallel work when a small batch would otherwise underfill the GPU.
    #[allow(clippy::manual_div_ceil)]
    #[cube(launch)]
    fn selective_scan_forward_parallel(
        delta: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        d: &Tensor<f32>,
        h_in: &Tensor<f32>,
        y: &mut Tensor<f32>,
        checkpoints: &mut Tensor<f32>,
        h_out: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] checkpoint_interval: usize,
        #[comptime] save_checkpoints: bool,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let n = UNIT_POS_X as usize;
        let local_channel = UNIT_POS_Y as usize;
        let batch = CUBE_POS_Y as usize;
        let channel = CUBE_POS_X as usize * FORWARD_CHANNELS as usize + local_channel;
        let active_channel = channel < channels;
        let active = active_channel && n < state_dim;
        let state_base = (batch * channels + channel) * state_dim;
        let a_base = channel * state_dim;
        let checkpoint_count = (seq_len + checkpoint_interval - 1) / checkpoint_interval;
        let mut state = if active { h_in[state_base + n] } else { 0.0f32 };

        for t in 0..seq_len {
            let btc = (batch * seq_len + t) * channels + channel;
            let btn = (batch * seq_len + t) * state_dim;
            let dt = if active_channel { delta[btc] } else { 0.0f32 };
            let x = if active_channel { xs[btc] } else { 0.0f32 };
            let contribution = if active {
                state = state * (dt * a[a_base + n]).exp() + dt * b_mat[btn + n] * x;
                state * c_mat[btn + n]
            } else {
                0.0f32
            };
            let out = plane_sum(contribution);
            if n == 0 && active_channel {
                y[btc] = out + x * d[channel];
            }

            if active
                && save_checkpoints
                && ((t + 1) % checkpoint_interval == 0 || t + 1 == seq_len)
            {
                let checkpoint = t / checkpoint_interval;
                let checkpoint_base =
                    ((batch * checkpoint_count + checkpoint) * channels + channel) * state_dim;
                checkpoints[checkpoint_base + n] = state;
            }
        }
        if active {
            h_out[state_base + n] = state;
        }
    }

    /// Per-segment forward transition from a zero entering state. One thread
    /// owns a `(batch, segment, channel)` triple, so every checkpoint segment
    /// of every scan is in flight at once instead of one thread walking the
    /// whole sequence. `decay` is `exp(A · Σdt)`, the exact product of the
    /// per-step decays of a diagonal state matrix.
    #[allow(clippy::manual_div_ceil)]
    #[cube(launch, fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
    fn selective_scan_forward_segment_partials(
        delta: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        partial: &mut Tensor<f32>,
        decay: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] segment_len: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let segments = (seq_len + segment_len - 1) / segment_len;
        let idx = ABSOLUTE_POS;
        if idx < partial.len() / state_dim {
            let channel = idx % channels;
            let segment = (idx / channels) % segments;
            let batch = idx / (channels * segments);
            let a_base = channel * state_dim;
            let start = segment * segment_len;
            let mut state = Array::<f32>::new(state_dim);
            let mut a_row = Array::<f32>::new(state_dim);
            #[unroll]
            for n in 0..state_dim {
                state[n] = 0.0f32;
                a_row[n] = a[a_base + n];
            }
            let mut sum_dt = 0.0f32;
            for i in 0..segment_len {
                let t = start + i;
                if t < seq_len {
                    let btc = (batch * seq_len + t) * channels + channel;
                    let btn = (batch * seq_len + t) * state_dim;
                    let dt = delta[btc];
                    let x = xs[btc];
                    sum_dt += dt;
                    #[unroll]
                    for n in 0..state_dim {
                        state[n] = state[n] * (dt * a_row[n]).exp() + dt * b_mat[btn + n] * x;
                    }
                }
            }
            let out_base = idx * state_dim;
            #[unroll]
            for n in 0..state_dim {
                partial[out_base + n] = state[n];
                decay[out_base + n] = (a_row[n] * sum_dt).exp();
            }
        }
    }

    /// Folds segment transitions serially into each segment's entering state.
    /// The fold touches `segments × state_dim` values per scan — a negligible
    /// stitch pass that unlocks segment-parallel recurrence kernels.
    #[cube(launch)]
    fn selective_scan_forward_segment_carry(
        h_in: &Tensor<f32>,
        partial: &Tensor<f32>,
        decay: &Tensor<f32>,
        carry: &mut Tensor<f32>,
        channels: u32,
        segments: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let segments = segments as usize;
        let idx = ABSOLUTE_POS;
        if idx < h_in.len() {
            let n = idx % state_dim;
            let bc = idx / state_dim;
            let channel = bc % channels;
            let batch = bc / channels;
            let mut state = h_in[idx];
            for s in 0..segments {
                let base = ((batch * segments + s) * channels + channel) * state_dim + n;
                carry[base] = state;
                state = partial[base] + decay[base] * state;
            }
        }
    }

    /// Re-runs each segment from its stitched entering state, producing the
    /// outputs, the per-segment checkpoints consumed by backward, and the
    /// final state.
    #[allow(clippy::manual_div_ceil)]
    #[cube(launch, fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
    fn selective_scan_forward_segment_apply(
        delta: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        d: &Tensor<f32>,
        carry: &Tensor<f32>,
        y: &mut Tensor<f32>,
        checkpoints: &mut Tensor<f32>,
        h_out: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] segment_len: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let segments = (seq_len + segment_len - 1) / segment_len;
        let idx = ABSOLUTE_POS;
        if idx < carry.len() / state_dim {
            let channel = idx % channels;
            let segment = (idx / channels) % segments;
            let batch = idx / (channels * segments);
            let a_base = channel * state_dim;
            let start = segment * segment_len;
            let carry_base = idx * state_dim;
            let mut state = Array::<f32>::new(state_dim);
            let mut a_row = Array::<f32>::new(state_dim);
            #[unroll]
            for n in 0..state_dim {
                state[n] = carry[carry_base + n];
                a_row[n] = a[a_base + n];
            }
            for i in 0..segment_len {
                let t = start + i;
                if t < seq_len {
                    let btc = (batch * seq_len + t) * channels + channel;
                    let btn = (batch * seq_len + t) * state_dim;
                    let dt = delta[btc];
                    let x = xs[btc];
                    let mut out = 0.0f32;
                    #[unroll]
                    for n in 0..state_dim {
                        state[n] = state[n] * (dt * a_row[n]).exp() + dt * b_mat[btn + n] * x;
                        out += state[n] * c_mat[btn + n];
                    }
                    y[btc] = out + x * d[channel];
                }
            }
            let checkpoint_base = ((batch * segments + segment) * channels + channel) * state_dim;
            #[unroll]
            for n in 0..state_dim {
                checkpoints[checkpoint_base + n] = state[n];
            }
            if segment + 1 == segments {
                let state_base = (batch * channels + channel) * state_dim;
                #[unroll]
                for n in 0..state_dim {
                    h_out[state_base + n] = state[n];
                }
            }
        }
    }

    /// Per-segment adjoint transition from a zero adjoint. The adjoint
    /// recurrence `adj_{t-1} = (adj_t + dy_t·C_t)·α_t` is linear in `adj`, so
    /// segments compose exactly like the forward pass, just right-to-left.
    #[allow(clippy::manual_div_ceil)]
    #[cube(launch, fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
    fn selective_scan_backward_segment_partials(
        delta: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        grad_y: &Tensor<f32>,
        inj: &mut Tensor<f32>,
        decay: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] segment_len: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let segments = (seq_len + segment_len - 1) / segment_len;
        let idx = ABSOLUTE_POS;
        if idx < inj.len() / state_dim {
            let channel = idx % channels;
            let segment = (idx / channels) % segments;
            let batch = idx / (channels * segments);
            let a_base = channel * state_dim;
            let start = segment * segment_len;
            let mut adj = Array::<f32>::new(state_dim);
            let mut a_row = Array::<f32>::new(state_dim);
            #[unroll]
            for n in 0..state_dim {
                adj[n] = 0.0f32;
                a_row[n] = a[a_base + n];
            }
            let mut sum_dt = 0.0f32;
            for i_rev in 0..segment_len {
                let i = segment_len - i_rev - 1;
                let t = start + i;
                if t < seq_len {
                    let btc = (batch * seq_len + t) * channels + channel;
                    let btn = (batch * seq_len + t) * state_dim;
                    let dt = delta[btc];
                    let dy = grad_y[btc];
                    sum_dt += dt;
                    #[unroll]
                    for n in 0..state_dim {
                        adj[n] = (adj[n] + dy * c_mat[btn + n]) * (dt * a_row[n]).exp();
                    }
                }
            }
            let out_base = idx * state_dim;
            #[unroll]
            for n in 0..state_dim {
                inj[out_base + n] = adj[n];
                decay[out_base + n] = (a_row[n] * sum_dt).exp();
            }
        }
    }

    /// Folds adjoint segment transitions from the sequence end backwards,
    /// producing the adjoint entering each segment from its right boundary.
    #[cube(launch)]
    fn selective_scan_backward_segment_carry(
        inj: &Tensor<f32>,
        decay: &Tensor<f32>,
        carry: &mut Tensor<f32>,
        channels: u32,
        segments: u32,
        batch_channels_states: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let segments = segments as usize;
        let idx = ABSOLUTE_POS;
        if idx < batch_channels_states as usize {
            let n = idx % state_dim;
            let bc = idx / state_dim;
            let channel = bc % channels;
            let batch = bc / channels;
            let mut adj = 0.0f32;
            for s_rev in 0..segments {
                let s = segments - s_rev - 1;
                let base = ((batch * segments + s) * channels + channel) * state_dim + n;
                carry[base] = adj;
                adj = inj[base] + decay[base] * adj;
            }
        }
    }

    /// One block owns a `(batch, channel tile)` and computes every scan
    /// gradient while following the reverse recurrence exactly once. The
    /// channel and state reductions share the per-token contributions in
    /// workgroup memory instead of launching a second recurrent kernel.
    #[allow(clippy::manual_div_ceil, clippy::useless_conversion)]
    #[cube(launch)]
    fn selective_scan_backward_fused(
        delta: &Tensor<f32>,
        delta_raw: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        d: &Tensor<f32>,
        h_in: &Tensor<f32>,
        checkpoints: &Tensor<f32>,
        grad_y: &Tensor<f32>,
        grad_delta: &mut Tensor<f32>,
        grad_xs: &mut Tensor<f32>,
        grad_b: &mut Tensor<Atomic<f32>>,
        grad_c: &mut Tensor<Atomic<f32>>,
        grad_a: &mut Tensor<Atomic<f32>>,
        grad_d: &mut Tensor<Atomic<f32>>,
        grad_h: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] checkpoint_interval: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let local_channel = UNIT_POS_X as usize;
        let n = UNIT_POS_Y as usize;
        let batch = CUBE_POS_Y as usize;
        let channel = CUBE_POS_X as usize * BACKWARD_CHANNELS + local_channel;
        let active_channel = channel < channels;
        let active = active_channel && n < state_dim;
        let state_index = (batch * channels + channel) * state_dim + n;
        let a_index = channel * state_dim + n;
        let shared_index = n * BACKWARD_CHANNELS + local_channel;
        let shared_len = BACKWARD_CHANNELS * state_dim;
        // Ping-pong buffers avoid a second barrier; the intervening step's
        // barrier completes before either buffer is reused.
        let mut grad_dt_shared = Shared::new_slice(2 * shared_len);
        let mut grad_x_shared = Shared::new_slice(2 * shared_len);
        let mut adjoint = 0.0f32;
        let mut grad_a_local = 0.0f32;
        let mut grad_d_local = 0.0f32;
        let checkpoint_count = (seq_len + checkpoint_interval - 1) / checkpoint_interval;

        for checkpoint_step in 0..checkpoint_count {
            let checkpoint = checkpoint_count - checkpoint_step - 1;
            let start = checkpoint * checkpoint_interval;
            let mut end = start + checkpoint_interval;
            if end > seq_len {
                end = seq_len;
            }
            let chunk_len = end - start;
            let state_before = if active {
                if checkpoint == 0 {
                    h_in[state_index]
                } else {
                    checkpoints[((batch * checkpoint_count + checkpoint - 1) * channels + channel)
                        * state_dim
                        + n]
                }
            } else {
                0.0f32
            };
            let av = if active { a[a_index] } else { 0.0f32 };
            let mut state = state_before;
            let mut chunk_states = Array::<f32>::new(checkpoint_interval);

            if checkpoint_interval == 1 {
                chunk_states[0] = if active {
                    checkpoints[((batch * checkpoint_count + checkpoint) * channels + channel)
                        * state_dim
                        + n]
                } else {
                    0.0f32
                };
            } else {
                for offset in 0..chunk_len {
                    let t = start + offset;
                    let btc = (batch * seq_len + t) * channels + channel;
                    let btn = (batch * seq_len + t) * state_dim;
                    let dt = if active { delta[btc] } else { 0.0f32 };
                    let x = if active { xs[btc] } else { 0.0f32 };
                    let bv = if active { b_mat[btn + n] } else { 0.0f32 };
                    state = state * (dt * av).exp() + dt * bv * x;
                    chunk_states[offset] = state;
                }
            }

            for reverse_offset in 0..chunk_len {
                let offset = chunk_len - reverse_offset - 1;
                let t = start + offset;
                let btc = (batch * seq_len + t) * channels + channel;
                let btn = (batch * seq_len + t) * state_dim;
                let dt = if active { delta[btc] } else { 0.0f32 };
                let raw_dt = if active { delta_raw[btc] } else { 0.0f32 };
                let x = if active { xs[btc] } else { 0.0f32 };
                let dy = if active { grad_y[btc] } else { 0.0f32 };
                let bv = if active { b_mat[btn + n] } else { 0.0f32 };
                let cv = if active { c_mat[btn + n] } else { 0.0f32 };
                let alpha = (dt * av).exp();
                let h_prev = if offset == 0 {
                    state_before
                } else {
                    chunk_states[offset - 1]
                };
                let h_t = chunk_states[offset];
                let g = adjoint + dy * cv;
                let step = seq_len - t - 1;
                let shared_offset = (step % 2) * shared_len;
                grad_dt_shared[shared_offset + shared_index] = g * (h_prev * alpha * av + bv * x);
                grad_x_shared[shared_offset + shared_index] = g * dt * bv;
                let grad_b_sum = half_plane_sum(g * dt * x, local_channel);
                let grad_c_sum = half_plane_sum(dy * h_t, local_channel);
                sync_cube();

                if n == 0 && active_channel {
                    let mut grad_dt = 0.0f32;
                    let mut grad_x = 0.0f32;
                    for state in 0..state_dim {
                        let index = shared_offset + state * BACKWARD_CHANNELS + local_channel;
                        grad_dt += grad_dt_shared[index];
                        grad_x += grad_x_shared[index];
                    }
                    grad_delta[btc] = grad_dt * stable_sigmoid(raw_dt);
                    grad_xs[btc] = dy * d[channel] + grad_x;
                    grad_d_local += dy * x;
                }
                if local_channel == 0 {
                    atomic_add_f32(&mut grad_b[btn + n], grad_b_sum);
                    atomic_add_f32(&mut grad_c[btn + n], grad_c_sum);
                }
                grad_a_local += g * h_prev * alpha * dt;
                adjoint = g * alpha;
            }
        }

        if active {
            atomic_add_f32(&mut grad_a[a_index], grad_a_local);
            grad_h[state_index] = adjoint;
        }
        if n == 0 && active_channel {
            atomic_add_f32(&mut grad_d[channel], grad_d_local);
        }
    }

    /// Segment-parallel variant of the fused backward kernel: one block owns a
    /// `(batch, channel tile, segment)` and re-derives its forward states from
    /// the segment checkpoint, while the adjoint entering from the right
    /// boundary comes from the stitched carry. All segments run concurrently
    /// instead of one block walking the whole reverse recurrence.
    #[allow(clippy::manual_div_ceil, clippy::useless_conversion)]
    #[cube(launch, fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
    fn selective_scan_backward_segmented(
        delta: &Tensor<f32>,
        delta_raw: &Tensor<f32>,
        xs: &Tensor<f32>,
        b_mat: &Tensor<f32>,
        c_mat: &Tensor<f32>,
        a: &Tensor<f32>,
        d: &Tensor<f32>,
        h_in: &Tensor<f32>,
        checkpoints: &Tensor<f32>,
        adj_carry: &Tensor<f32>,
        grad_y: &Tensor<f32>,
        grad_delta: &mut Tensor<f32>,
        grad_xs: &mut Tensor<f32>,
        grad_b: &mut Tensor<Atomic<f32>>,
        grad_c: &mut Tensor<Atomic<f32>>,
        grad_a: &mut Tensor<Atomic<f32>>,
        grad_d: &mut Tensor<Atomic<f32>>,
        grad_h: &mut Tensor<f32>,
        channels: u32,
        seq_len: u32,
        #[comptime] state_dim: usize,
        #[comptime] segment_len: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let local_channel = UNIT_POS_X as usize;
        let n = UNIT_POS_Y as usize;
        let batch = CUBE_POS_Y as usize;
        let segment = CUBE_POS_Z as usize;
        let channel = CUBE_POS_X as usize * BACKWARD_CHANNELS + local_channel;
        let active_channel = channel < channels;
        let active = active_channel && n < state_dim;
        let state_index = (batch * channels + channel) * state_dim + n;
        let a_index = channel * state_dim + n;
        let shared_index = n * BACKWARD_CHANNELS + local_channel;
        let shared_len = BACKWARD_CHANNELS * state_dim;
        // Ping-pong buffers avoid a second barrier; the intervening step's
        // barrier completes before either buffer is reused.
        let mut grad_dt_shared = Shared::new_slice(2 * shared_len);
        let mut grad_x_shared = Shared::new_slice(2 * shared_len);
        let mut grad_a_local = 0.0f32;
        let mut grad_d_local = 0.0f32;
        let segments = (seq_len + segment_len - 1) / segment_len;
        let segment_base = ((batch * segments + segment) * channels + channel) * state_dim + n;
        let mut adjoint = if active {
            adj_carry[segment_base]
        } else {
            0.0f32
        };

        let start = segment * segment_len;
        let mut end = start + segment_len;
        if end > seq_len {
            end = seq_len;
        }
        let chunk_len = end - start;
        let state_before = if active {
            if segment == 0 {
                h_in[state_index]
            } else {
                checkpoints[((batch * segments + segment - 1) * channels + channel) * state_dim + n]
            }
        } else {
            0.0f32
        };
        let av = if active { a[a_index] } else { 0.0f32 };
        let mut state = state_before;
        let mut chunk_states = Array::<f32>::new(segment_len);

        for offset in 0..chunk_len {
            let t = start + offset;
            let btc = (batch * seq_len + t) * channels + channel;
            let btn = (batch * seq_len + t) * state_dim;
            let dt = if active { delta[btc] } else { 0.0f32 };
            let x = if active { xs[btc] } else { 0.0f32 };
            let bv = if active { b_mat[btn + n] } else { 0.0f32 };
            state = state * (dt * av).exp() + dt * bv * x;
            chunk_states[offset] = state;
        }

        for reverse_offset in 0..chunk_len {
            let offset = chunk_len - reverse_offset - 1;
            let t = start + offset;
            let btc = (batch * seq_len + t) * channels + channel;
            let btn = (batch * seq_len + t) * state_dim;
            let dt = if active { delta[btc] } else { 0.0f32 };
            let raw_dt = if active { delta_raw[btc] } else { 0.0f32 };
            let x = if active { xs[btc] } else { 0.0f32 };
            let dy = if active { grad_y[btc] } else { 0.0f32 };
            let bv = if active { b_mat[btn + n] } else { 0.0f32 };
            let cv = if active { c_mat[btn + n] } else { 0.0f32 };
            let alpha = (dt * av).exp();
            let h_prev = if offset == 0 {
                state_before
            } else {
                chunk_states[offset - 1]
            };
            let h_t = chunk_states[offset];
            let g = adjoint + dy * cv;
            let step = seq_len - t - 1;
            let shared_offset = (step % 2) * shared_len;
            grad_dt_shared[shared_offset + shared_index] = g * (h_prev * alpha * av + bv * x);
            grad_x_shared[shared_offset + shared_index] = g * dt * bv;
            let grad_b_sum = half_plane_sum(g * dt * x, local_channel);
            let grad_c_sum = half_plane_sum(dy * h_t, local_channel);
            sync_cube();

            if n == 0 && active_channel {
                let mut grad_dt = 0.0f32;
                let mut grad_x = 0.0f32;
                for state in 0..state_dim {
                    let index = shared_offset + state * BACKWARD_CHANNELS + local_channel;
                    grad_dt += grad_dt_shared[index];
                    grad_x += grad_x_shared[index];
                }
                grad_delta[btc] = grad_dt * stable_sigmoid(raw_dt);
                grad_xs[btc] = dy * d[channel] + grad_x;
                grad_d_local += dy * x;
            }
            if local_channel == 0 {
                atomic_add_f32(&mut grad_b[btn + n], grad_b_sum);
                atomic_add_f32(&mut grad_c[btn + n], grad_c_sum);
            }
            grad_a_local += g * h_prev * alpha * dt;
            adjoint = g * alpha;
        }

        if active {
            atomic_add_f32(&mut grad_a[a_index], grad_a_local);
            if segment == 0 {
                grad_h[state_index] = adjoint;
            }
        }
        if n == 0 && active_channel {
            atomic_add_f32(&mut grad_d[channel], grad_d_local);
        }
    }

    impl<R: CubeRuntime> MambaBackend for CubeBackend<R> {
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
        ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
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
            let y = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let h_out = empty_like(&xs, Shape::new([batch, channels, state_dim]));
            let delta = empty_like(&xs, Shape::new([batch, seq_len, channels]));
            let checkpoint_interval = scan_checkpoint_interval(batch, seq_len, channels, state_dim);
            let full_sequence_scan = save_states || seq_len > 1;
            let states = if save_states {
                empty_like(
                    &xs,
                    Shape::new([
                        batch,
                        seq_len.div_ceil(checkpoint_interval),
                        channels,
                        state_dim,
                    ]),
                )
            } else {
                h_out.clone()
            };

            let client = xs.client.clone();
            let delta_total = (batch * seq_len * channels) as u32;
            softplus_forward::launch::<R>(
                &client,
                CubeCount::Static(delta_total.div_ceil(THREADS_PER_CUBE), 1, 1),
                CubeDim::new_1d(THREADS_PER_CUBE),
                delta_raw.into_tensor_arg(),
                delta.clone().into_tensor_arg(),
            );
            let segments = seq_len.div_ceil(checkpoint_interval);
            if full_sequence_scan && save_states && checkpoint_interval > 1 && segments > 1 {
                // Training path: segment-parallel scan. Segment transitions
                // compose exactly for a diagonal recurrence, so every
                // checkpoint segment runs concurrently and a cheap fold
                // stitches the carries.
                let partial_shape = Shape::new([batch, segments, channels, state_dim]);
                let partial = empty_like(&xs, partial_shape.clone());
                let decay = empty_like(&xs, partial_shape.clone());
                let carry = empty_like(&xs, partial_shape);
                let segment_threads = (batch * segments * channels) as u32;
                selective_scan_forward_segment_partials::launch::<R>(
                    &client,
                    CubeCount::Static(segment_threads.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.clone().into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.clone().into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    partial.clone().into_tensor_arg(),
                    decay.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                );
                let carry_threads = (batch * channels * state_dim) as u32;
                selective_scan_forward_segment_carry::launch::<R>(
                    &client,
                    CubeCount::Static(carry_threads.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    h.clone().into_tensor_arg(),
                    partial.into_tensor_arg(),
                    decay.into_tensor_arg(),
                    carry.clone().into_tensor_arg(),
                    channels as u32,
                    segments as u32,
                    state_dim,
                );
                selective_scan_forward_segment_apply::launch::<R>(
                    &client,
                    CubeCount::Static(segment_threads.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.clone().into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.clone().into_tensor_arg(),
                    c_mat.into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    d.into_tensor_arg(),
                    carry.into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    states.clone().into_tensor_arg(),
                    h_out.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                );
            } else if full_sequence_scan {
                let serial_blocks = ((batch * channels) as u32).div_ceil(THREADS_PER_CUBE);
                if serial_blocks >= SERIAL_SCAN_MIN_BLOCKS {
                    selective_scan_forward_serial::launch::<R>(
                        &client,
                        CubeCount::Static(serial_blocks, 1, 1),
                        CubeDim::new_1d(THREADS_PER_CUBE),
                        delta.clone().into_tensor_arg(),
                        xs.clone().into_tensor_arg(),
                        b_mat.clone().into_tensor_arg(),
                        c_mat.into_tensor_arg(),
                        a.clone().into_tensor_arg(),
                        d.into_tensor_arg(),
                        h.clone().into_tensor_arg(),
                        y.clone().into_tensor_arg(),
                        states.clone().into_tensor_arg(),
                        h_out.clone().into_tensor_arg(),
                        channels as u32,
                        seq_len as u32,
                        state_dim,
                        checkpoint_interval,
                        save_states,
                    );
                } else {
                    selective_scan_forward_parallel::launch::<R>(
                        &client,
                        CubeCount::Static(
                            (channels as u32).div_ceil(FORWARD_CHANNELS),
                            batch as u32,
                            1,
                        ),
                        CubeDim::new_2d(PLANE_WIDTH, FORWARD_CHANNELS),
                        delta.clone().into_tensor_arg(),
                        xs.clone().into_tensor_arg(),
                        b_mat.clone().into_tensor_arg(),
                        c_mat.into_tensor_arg(),
                        a.clone().into_tensor_arg(),
                        d.into_tensor_arg(),
                        h.clone().into_tensor_arg(),
                        y.clone().into_tensor_arg(),
                        states.clone().into_tensor_arg(),
                        h_out.clone().into_tensor_arg(),
                        channels as u32,
                        seq_len as u32,
                        state_dim,
                        checkpoint_interval,
                        save_states,
                    );
                }
            } else {
                let total = (batch * channels) as u32;
                selective_scan_step::launch::<R>(
                    &client,
                    CubeCount::Static(total.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.into_tensor_arg(),
                    c_mat.into_tensor_arg(),
                    a.into_tensor_arg(),
                    d.into_tensor_arg(),
                    h.into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    h_out.clone().into_tensor_arg(),
                    channels as u32,
                    state_dim,
                );
            }

            (y, h_out, states)
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
        ) -> (
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
        ) {
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
            let checkpoint_interval = scan_checkpoint_interval(batch, seq_len, channels, state_dim);
            let client = xs.client.clone();

            let delta_total = (batch * seq_len * channels) as u32;
            softplus_forward::launch::<R>(
                &client,
                CubeCount::Static(delta_total.div_ceil(THREADS_PER_CUBE), 1, 1),
                CubeDim::new_1d(THREADS_PER_CUBE),
                delta_raw.clone().into_tensor_arg(),
                delta.clone().into_tensor_arg(),
            );

            let segments = seq_len.div_ceil(checkpoint_interval);
            if checkpoint_interval > 1 && segments > 1 {
                // Segment-parallel adjoint: stitch the linear adjoint
                // recurrence across checkpoint segments, then run every
                // segment's gradient block concurrently.
                let partial_shape = Shape::new([batch, segments, channels, state_dim]);
                let inj = empty_like(&xs, partial_shape.clone());
                let adecay = empty_like(&xs, partial_shape.clone());
                let acarry = empty_like(&xs, partial_shape);
                let segment_threads = (batch * segments * channels) as u32;
                selective_scan_backward_segment_partials::launch::<R>(
                    &client,
                    CubeCount::Static(segment_threads.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.clone().into_tensor_arg(),
                    c_mat.clone().into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    grad_y.clone().into_tensor_arg(),
                    inj.clone().into_tensor_arg(),
                    adecay.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                );
                let carry_threads = (batch * channels * state_dim) as u32;
                selective_scan_backward_segment_carry::launch::<R>(
                    &client,
                    CubeCount::Static(carry_threads.div_ceil(THREADS_PER_CUBE), 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    inj.into_tensor_arg(),
                    adecay.into_tensor_arg(),
                    acarry.clone().into_tensor_arg(),
                    channels as u32,
                    segments as u32,
                    carry_threads,
                    state_dim,
                );
                selective_scan_backward_segmented::launch::<R>(
                    &client,
                    CubeCount::Static(
                        (channels as u32).div_ceil(BACKWARD_CHANNELS as u32),
                        batch as u32,
                        segments as u32,
                    ),
                    CubeDim::new_2d(BACKWARD_CHANNELS as u32, state_dim as u32),
                    delta.clone().into_tensor_arg(),
                    delta_raw.into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.into_tensor_arg(),
                    c_mat.clone().into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    d.into_tensor_arg(),
                    h.clone().into_tensor_arg(),
                    states.clone().into_tensor_arg(),
                    acarry.into_tensor_arg(),
                    grad_y.clone().into_tensor_arg(),
                    grad_delta.clone().into_tensor_arg(),
                    grad_xs.clone().into_tensor_arg(),
                    grad_b.clone().into_tensor_arg(),
                    grad_c.clone().into_tensor_arg(),
                    grad_a.clone().into_tensor_arg(),
                    grad_d.clone().into_tensor_arg(),
                    grad_h.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                );
            } else {
                selective_scan_backward_fused::launch::<R>(
                    &client,
                    CubeCount::Static(
                        (channels as u32).div_ceil(BACKWARD_CHANNELS as u32),
                        batch as u32,
                        1,
                    ),
                    CubeDim::new_2d(BACKWARD_CHANNELS as u32, state_dim as u32),
                    delta.clone().into_tensor_arg(),
                    delta_raw.into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.into_tensor_arg(),
                    c_mat.clone().into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    d.into_tensor_arg(),
                    h.clone().into_tensor_arg(),
                    states.clone().into_tensor_arg(),
                    grad_y.clone().into_tensor_arg(),
                    grad_delta.clone().into_tensor_arg(),
                    grad_xs.clone().into_tensor_arg(),
                    grad_b.clone().into_tensor_arg(),
                    grad_c.clone().into_tensor_arg(),
                    grad_a.clone().into_tensor_arg(),
                    grad_d.clone().into_tensor_arg(),
                    grad_h.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                );
            }

            (grad_delta, grad_xs, grad_b, grad_c, grad_a, grad_d, grad_h)
        }
    }
}

#[cfg(test)]
mod policy_tests {
    use super::{CHECKPOINTED_SCAN_INTERVAL, scan_checkpoint_interval};

    #[test]
    fn full_states_are_limited_to_small_bounded_batches() {
        assert_eq!(scan_checkpoint_interval(2, 4096, 1024, 16), 1);
        assert_eq!(
            scan_checkpoint_interval(16, 1024, 1024, 16),
            CHECKPOINTED_SCAN_INTERVAL
        );
        assert_eq!(
            scan_checkpoint_interval(2, 8192, 1024, 16),
            CHECKPOINTED_SCAN_INTERVAL
        );
    }
}

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::selective_scan;
    use crate::model::test_support::{max_diff, values};

    fn gpu_device() -> Device {
        #[cfg(feature = "metal")]
        return Device::metal(burn::tensor::DeviceKind::DefaultDevice);

        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        return Device::cuda(0);
    }

    #[test]
    fn test_cubecl_selective_scan_matches_ndarray_reference() {
        let cpu = Device::ndarray();
        let gpu = gpu_device();
        let (batch, seq_len, channels, state_dim) = (2, 35, 3, 4);

        let delta = values(batch * seq_len * channels, 0.13, 0.08);
        let xs = values(batch * seq_len * channels, 0.17, -0.03);
        let b_mat = values(batch * seq_len * state_dim, 0.19, 0.02);
        let c_mat = values(batch * seq_len * state_dim, 0.23, -0.01);
        let a = values(channels * state_dim, 0.11, -0.4);
        let d = values(channels, 0.07, 0.9);
        let h = values(batch * channels * state_dim, 0.05, 0.01);

        macro_rules! tensor {
            ($device:expr, $data:expr, $shape:expr) => {
                Tensor::from_data(TensorData::new($data.clone(), $shape), $device)
            };
        }

        let (cpu_y, cpu_h) = selective_scan(
            tensor!(&cpu, delta, [batch, seq_len, channels]),
            tensor!(&cpu, xs, [batch, seq_len, channels]),
            tensor!(&cpu, b_mat, [batch, seq_len, state_dim]),
            tensor!(&cpu, c_mat, [batch, seq_len, state_dim]),
            tensor!(&cpu, a, [channels, state_dim]),
            tensor!(&cpu, d, [channels]),
            tensor!(&cpu, h, [batch, channels, state_dim]),
            state_dim,
        );
        let (gpu_y, gpu_h) = selective_scan(
            tensor!(&gpu, delta, [batch, seq_len, channels]),
            tensor!(&gpu, xs, [batch, seq_len, channels]),
            tensor!(&gpu, b_mat, [batch, seq_len, state_dim]),
            tensor!(&gpu, c_mat, [batch, seq_len, state_dim]),
            tensor!(&gpu, a, [channels, state_dim]),
            tensor!(&gpu, d, [channels]),
            tensor!(&gpu, h, [batch, channels, state_dim]),
            state_dim,
        );

        assert!(max_diff(cpu_y.into_data(), gpu_y.into_data()) < 1e-5);
        assert!(max_diff(cpu_h.into_data(), gpu_h.into_data()) < 1e-5);
    }

    fn assert_cubecl_selective_scan_backward(batch: usize) {
        let cpu = Device::ndarray().autodiff();
        let gpu = gpu_device().autodiff();
        let (seq_len, channels, state_dim) = (37, 3, 4);
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
            ($device:expr) => {{
                let delta =
                    Tensor::<3>::from_data(TensorData::new(data.0.clone(), shapes.0), $device)
                        .require_grad();
                let xs = Tensor::<3>::from_data(TensorData::new(data.1.clone(), shapes.0), $device)
                    .require_grad();
                let b = Tensor::<3>::from_data(TensorData::new(data.2.clone(), shapes.1), $device)
                    .require_grad();
                let c = Tensor::<3>::from_data(TensorData::new(data.3.clone(), shapes.1), $device)
                    .require_grad();
                let a = Tensor::<2>::from_data(TensorData::new(data.4.clone(), shapes.2), $device)
                    .require_grad();
                let d = Tensor::<1>::from_data(TensorData::new(data.5.clone(), shapes.3), $device)
                    .require_grad();
                let h = Tensor::<3>::from_data(TensorData::new(data.6.clone(), shapes.4), $device)
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
                let weights = Tensor::<3>::from_data(
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

        let cpu_grads = run!(&cpu);
        let gpu_grads = run!(&gpu);
        for ((name, cpu), gpu) in ["delta", "xs", "b", "c", "a", "d", "h"]
            .into_iter()
            .zip(cpu_grads)
            .zip(gpu_grads)
        {
            let difference = max_diff(cpu, gpu);
            assert!(difference < 2e-4, "{name} gradient max diff: {difference}");
        }
    }

    #[test]
    fn test_cubecl_selective_scan_full_state_backward_matches_ndarray() {
        assert_cubecl_selective_scan_backward(2);
    }

    #[test]
    fn test_cubecl_selective_scan_checkpointed_backward_matches_ndarray() {
        assert_cubecl_selective_scan_backward(3);
    }
}
