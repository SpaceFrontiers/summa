//! Differentiable tensor-op reference for the selective scan.
//!
//! This is the correctness oracle: it runs on any backend, autodiff can
//! differentiate straight through the forward, and the hand-written
//! backward mirrors the GPU kernels' math exactly (softplus folded in,
//! per-step states retained). The CPU backend implements [`MambaBackend`]
//! with it; the GPU parity tests compare against it.

use burn::backend::{Backend, DispatchKindConversion, DispatchTensor, tensor::FloatTensor};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use super::{MambaBackend, stable_softplus};

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
pub(super) fn reference_selective_scan_tensor(
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
