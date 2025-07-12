use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};

use crate::mal::{ModelDef, SsmDef};

/// Numerically stable softplus: max(x, 0) + ln(1 + exp(-|x|))
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let neg_abs = x.abs()?.neg()?;
    relu + (neg_abs.exp()? + 1.0)?.log()?
}

/// Recurrent state for one Mamba layer: the conv tail (last k-1 inputs) and
/// the SSM hidden state — the compact persistent memory, O(1) in sequence
/// length.
pub struct MambaState {
    /// [B, d_inner, conv_kernel - 1]
    pub conv: Tensor,
    /// [B, d_inner, state_dim]
    pub h: Tensor,
}

/// Selective state-space (Mamba-1) mixer.
///
/// Sequential scan implementation — correct and cache-free, sized for
/// inference; training uses the fused kernels on the hermes-train side.
/// Tensor names follow the mamba-ssm convention under `layers.{i}.ssm.*`.
pub struct MambaMixer {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    pub(crate) d_inner: usize,
    pub(crate) state_dim: usize,
    dt_rank: usize,
    pub(crate) conv_kernel: usize,
}

impl MambaMixer {
    pub fn new(config: &ModelDef, ssm: &SsmDef, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let d_inner = ssm.expand * hidden;
        let state_dim = ssm.state_dim;
        let dt_rank = config.dt_rank(ssm);
        let conv_kernel = ssm.conv_kernel;

        let in_proj = linear_no_bias(hidden, 2 * d_inner, vb.pp("in_proj"))?;
        let conv_vb = vb.pp("conv1d");
        let conv1d_weight = conv_vb.get_with_hints(
            (d_inner, 1, conv_kernel),
            "weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: (1.0 / conv_kernel as f64).sqrt(),
            },
        )?;
        let conv1d_bias = conv_vb.get_with_hints(d_inner, "bias", candle_nn::Init::Const(0.0))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + 2 * state_dim, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        // A_log/D are always overwritten by checkpoint load (inference-only
        // path); the real S4D init lives on the hermes-train side.
        let a_log =
            vb.get_with_hints((d_inner, state_dim), "A_log", candle_nn::Init::Const(0.0))?;
        let d = vb.get_with_hints(d_inner, "D", candle_nn::Init::Const(1.0))?;
        let out_proj = linear_no_bias(d_inner, hidden, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            d_inner,
            state_dim,
            dt_rank,
            conv_kernel,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_state(x, None)
    }

    /// Forward with optional recurrent state. With state, the conv window and
    /// SSM hidden state continue from previous calls and are updated in place
    /// — O(1) per token, independent of history length.
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        mut state: Option<&mut MambaState>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let use_fused = crate::metal_scan::fused_supported(x.device(), x.dtype(), self.state_dim);

        // Input projection → x, z gates
        let xz = self.in_proj.forward(x)?; // [B, L, 2*di]
        let xs = xz.narrow(2, 0, self.d_inner)?;
        let z = xz.narrow(2, self.d_inner, self.d_inner)?;

        // Depthwise causal conv over time: [B, di, L]. Left context is the
        // stored conv tail when stateful, zeros otherwise.
        let xs_t = xs.transpose(1, 2)?.contiguous()?; // [B, di, L]
        let padded = match &state {
            Some(s) => Tensor::cat(&[&s.conv, &xs_t], 2)?.contiguous()?,
            None => xs_t.pad_with_zeros(2, self.conv_kernel - 1, 0)?,
        };
        if let Some(s) = state.as_deref_mut() {
            // New tail: last conv_kernel-1 raw inputs
            let total = padded.dim(2)?;
            s.conv = padded
                .narrow(2, total - (self.conv_kernel - 1), self.conv_kernel - 1)?
                .contiguous()?;
        }
        let conv = if use_fused {
            // Candle lowers grouped conv1d to one conv per group — ~16k
            // dispatches per decoded token with groups == d_inner (measured
            // 98% of Metal decode time). Fused: one dispatch, bias included.
            let op = crate::metal_scan::FusedDepthwiseConv1d {
                batch,
                d_inner: self.d_inner,
                l_in: padded.dim(2)?,
                kernel: self.conv_kernel,
            };
            padded.apply_op3_no_bwd(
                &self.conv1d_weight.flatten_all()?.contiguous()?,
                &self.conv1d_bias,
                &op,
            )?
        } else {
            let conv = padded.conv1d(&self.conv1d_weight, 0, 1, 1, self.d_inner)?;
            conv.broadcast_add(&self.conv1d_bias.reshape((1, self.d_inner, 1))?)?
        };
        let conv = conv.narrow(2, 0, seq_len)?;
        let xs = candle_nn::ops::silu(&conv.transpose(1, 2)?.contiguous()?)?; // [B, L, di]

        // Input-dependent Δ, B, C. The narrows are made contiguous: Metal's
        // matmul rejects strided views (dt_proj on the Δ slice).
        let x_dbl = self.x_proj.forward(&xs)?; // [B, L, dt_rank + 2N]
        let delta = x_dbl.narrow(2, 0, self.dt_rank)?.contiguous()?;
        let b_mat = x_dbl
            .narrow(2, self.dt_rank, self.state_dim)?
            .contiguous()?; // [B, L, N]
        let c_mat = x_dbl
            .narrow(2, self.dt_rank + self.state_dim, self.state_dim)?
            .contiguous()?;
        let delta = softplus(&self.dt_proj.forward(&delta)?)?; // [B, L, di]

        let a = self.a_log.exp()?.neg()?; // [di, N]

        let h = match &state {
            Some(s) => s.h.clone(),
            None => Tensor::zeros((batch, self.d_inner, self.state_dim), x.dtype(), x.device())?,
        };
        let (y, h) = if use_fused {
            // Single-dispatch fused scan (docs/metal-selective-scan.md);
            // inputs/outputs packed to fit CustomOp3's 3-in/1-out shape.
            let xd = Tensor::cat(&[&xs, &delta], 2)?.contiguous()?;
            let bc = Tensor::cat(&[&b_mat, &c_mat], 2)?.contiguous()?;
            let adh =
                Tensor::cat(&[&a.flatten_all()?, &self.d, &h.flatten_all()?], 0)?.contiguous()?;
            let op = crate::metal_scan::FusedSelectiveScan {
                batch,
                seq_len,
                d_inner: self.d_inner,
                state_dim: self.state_dim,
            };
            let out = xd.apply_op3_no_bwd(&bc, &adh, &op)?;
            let y_len = batch * seq_len * self.d_inner;
            let y = out
                .narrow(0, 0, y_len)?
                .reshape((batch, seq_len, self.d_inner))?;
            let h = out
                .narrow(0, y_len, batch * self.d_inner * self.state_dim)?
                .reshape((batch, self.d_inner, self.state_dim))?;
            (y, h)
        } else {
            // Reference per-timestep scan (CUDA / non-f32 / oversized state)
            let mut h = h;
            let mut ys = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let dt = delta.narrow(1, t, 1)?.squeeze(1)?; // [B, di]
                let xt = xs.narrow(1, t, 1)?.squeeze(1)?; // [B, di]
                let bt = b_mat.narrow(1, t, 1)?.squeeze(1)?; // [B, N]
                let ct = c_mat.narrow(1, t, 1)?.squeeze(1)?; // [B, N]

                let dt_e = dt.unsqueeze(2)?; // [B, di, 1]
                let da = dt_e.broadcast_mul(&a.unsqueeze(0)?)?.exp()?; // [B, di, N]
                let dbx = dt_e
                    .broadcast_mul(&bt.unsqueeze(1)?)? // [B, di, N]
                    .broadcast_mul(&xt.unsqueeze(2)?)?;
                h = (h.mul(&da)? + dbx)?;

                let yt = h.broadcast_mul(&ct.unsqueeze(1)?)?.sum(2)?; // [B, di]
                let yt = (yt + xt.broadcast_mul(&self.d.unsqueeze(0)?)?)?;
                ys.push(yt.unsqueeze(1)?);
            }
            (Tensor::cat(&ys, 1)?, h) // [B, L, di]
        };

        // Persist the final SSM state
        if let Some(s) = state {
            s.h = h;
        }

        // Output gate + projection
        let y = y.mul(&candle_nn::ops::silu(&z)?)?;
        self.out_proj.forward(&y)
    }
}
