//! Mamba-1 selective state-space mixer.
//!
//! ndarray uses the readable tensor-op reference; Metal and CUDA dispatch the
//! same stateful CubeCL kernel directly on Burn's resident GPU tensors.

use burn::module::{Initializer, Param};
use burn::prelude::*;
use burn::tensor::activation::silu;
use burn_nn::conv::{Conv1d, Conv1dConfig};
use burn_nn::{Linear, LinearConfig};

use crate::mal::{ModelDef, SsmDef};

use super::conv::depthwise_conv1d;
use super::matmul::{linear_low_precision, prepare_linear_for_inference};
use super::scan::selective_scan;

/// Recurrent state for one Mamba layer.
#[derive(Debug, Clone)]
pub struct MambaState {
    /// Last `conv_kernel - 1` raw projected inputs, `[B, d_inner, K-1]`.
    pub conv: Tensor<3>,
    /// Selective SSM hidden state, `[B, d_inner, state_dim]`.
    pub h: Tensor<3>,
}

#[derive(Module, Debug)]
pub struct MambaMixer {
    in_proj: Linear,
    conv1d: Conv1d,
    x_proj: Linear,
    dt_proj: Linear,
    /// Logarithm of the continuous-time state matrix.
    a_log: Param<Tensor<2>>,
    /// Direct residual term in the state-space output.
    d: Param<Tensor<1>>,
    out_proj: Linear,
    #[module(skip)]
    pub(crate) d_inner: usize,
    #[module(skip)]
    pub(crate) state_dim: usize,
    #[module(skip)]
    dt_rank: usize,
    #[module(skip)]
    pub(crate) conv_kernel: usize,
}

impl MambaMixer {
    pub fn new(config: &ModelDef, ssm: &SsmDef, device: &Device) -> Self {
        assert!(ssm.expand > 0, "Mamba expand must be positive");
        assert!(ssm.state_dim > 0, "Mamba state_dim must be positive");
        assert!(ssm.conv_kernel > 0, "Mamba conv_kernel must be positive");

        let hidden = config.hidden_size;
        let d_inner = ssm.expand * hidden;
        let state_dim = ssm.state_dim;
        let dt_rank = config.dt_rank(ssm);
        let conv_kernel = ssm.conv_kernel;

        let linear =
            |d_in, d_out, bias| LinearConfig::new(d_in, d_out).with_bias(bias).init(device);
        let conv1d = Conv1dConfig::new(d_inner, d_inner, conv_kernel)
            .with_groups(d_inner)
            .with_bias(true)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: (1.0 / conv_kernel as f64).sqrt(),
            })
            .init(device);

        Self {
            in_proj: linear(hidden, 2 * d_inner, false),
            conv1d,
            x_proj: linear(d_inner, dt_rank + 2 * state_dim, false),
            dt_proj: linear(dt_rank, d_inner, true),
            a_log: Initializer::Zeros.init([d_inner, state_dim], device),
            d: Initializer::Ones.init([d_inner], device),
            out_proj: linear(d_inner, hidden, false),
            d_inner,
            state_dim,
            dt_rank,
            conv_kernel,
        }
    }

    pub fn make_state(&self, batch: usize, device: &Device) -> MambaState {
        assert!(batch > 0, "Mamba state batch size must be positive");
        MambaState {
            conv: Tensor::zeros([batch, self.d_inner, self.conv_kernel - 1], device),
            h: Tensor::zeros([batch, self.d_inner, self.state_dim], device),
        }
    }

    pub(crate) fn prepare_inference(&mut self) {
        for layer in [
            &mut self.in_proj,
            &mut self.x_proj,
            &mut self.dt_proj,
            &mut self.out_proj,
        ] {
            prepare_linear_for_inference(layer);
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_with_state(x, None)
    }

    /// Run the mixer and optionally continue/update its recurrent state.
    pub fn forward_with_state(
        &self,
        x: Tensor<3>,
        mut state: Option<&mut MambaState>,
    ) -> Tensor<3> {
        let [batch, seq_len, _] = x.dims();
        assert!(seq_len > 0, "Mamba forward requires at least one token");

        // The projection chain feeding the scan stays in the matmul compute
        // dtype (BF16 on CUDA training): the fused conv/scan kernels read it
        // directly, which removes the promote-and-recast passes around every
        // SSM layer and halves their sequence-tensor traffic.
        let xz = linear_low_precision(&self.in_proj, x);
        let xs = xz.clone().slice([0..batch, 0..seq_len, 0..self.d_inner]);
        let z = xz.slice([0..batch, 0..seq_len, self.d_inner..2 * self.d_inner]);

        // Burn Conv1d uses cross-correlation in both training and inference.
        let xs_t = xs.swap_dims(1, 2);
        let padded = match state.as_deref() {
            Some(s) => {
                assert_eq!(
                    s.conv.dims(),
                    [batch, self.d_inner, self.conv_kernel - 1],
                    "Mamba conv state shape does not match this input/layer"
                );
                assert_eq!(
                    s.h.dims(),
                    [batch, self.d_inner, self.state_dim],
                    "Mamba SSM state shape does not match this input/layer"
                );
                Tensor::cat(vec![s.conv.clone(), xs_t], 2)
            }
            None => {
                let mut padding: Tensor<3> =
                    Tensor::zeros([batch, self.d_inner, self.conv_kernel - 1], &xs_t.device());
                if padding.dtype() != xs_t.dtype() {
                    padding = padding.cast(burn::tensor::FloatDType::BF16);
                }
                Tensor::cat(vec![padding, xs_t], 2)
            }
        };
        if let Some(s) = state.as_deref_mut() {
            let total = padded.dims()[2];
            s.conv = padded.clone().slice([
                0..batch,
                0..self.d_inner,
                total - (self.conv_kernel - 1)..total,
            ]);
        }
        let xs = silu(
            depthwise_conv1d(
                padded,
                self.conv1d.weight.val(),
                self.conv1d
                    .bias
                    .as_ref()
                    .expect("Mamba depthwise convolution always has a bias")
                    .val(),
            )
            .swap_dims(1, 2),
        );

        // Input-dependent delta, B, C.
        let x_dbl = linear_low_precision(&self.x_proj, xs.clone());
        let delta = x_dbl.clone().slice([0..batch, 0..seq_len, 0..self.dt_rank]);
        let b_mat = x_dbl.clone().slice([
            0..batch,
            0..seq_len,
            self.dt_rank..self.dt_rank + self.state_dim,
        ]);
        let c_mat = x_dbl.slice([
            0..batch,
            0..seq_len,
            self.dt_rank + self.state_dim..self.dt_rank + 2 * self.state_dim,
        ]);

        let delta_raw = linear_low_precision(&self.dt_proj, delta);
        let a = -self.a_log.val().exp();

        let h = match state.as_deref() {
            Some(s) => s.h.clone(),
            None => Tensor::zeros([batch, self.d_inner, self.state_dim], &xs.device()),
        };
        // Softplus and its derivative are part of the scan kernel, avoiding a
        // chain of full-sequence elementwise launches around every SSM layer.
        let (y, h) = selective_scan(
            delta_raw,
            xs,
            b_mat,
            c_mat,
            a,
            self.d.val(),
            h,
            self.state_dim,
        );
        if let Some(s) = state {
            s.h = h;
        }

        linear_low_precision(&self.out_proj, y * silu(z))
    }
}
