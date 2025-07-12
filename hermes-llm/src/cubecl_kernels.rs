//! Portable Mamba-1 inference kernels written once in CubeCL and JIT-compiled
//! to CUDA / Metal(wgpu) / (future) CPU.
//!
//! This is an *added* path, gated behind the OFF-by-default `cubecl` feature.
//! The Candle sequential scan in [`crate::model::MambaMixer`] remains the
//! default and is the parity reference these kernels are checked against
//! (`tolerance < 1e-4`). See `docs/cubecl-migration.md`.
//!
//! Two kernels, mirroring the two compute-heavy steps of the Candle mixer:
//!
//! 1. [`conv1d_depthwise`] — depthwise **causal** conv1d + bias. One thread per
//!    `(batch, channel)`, looping over time. Matches Candle's
//!    `padded.conv1d(weight, 0, 1, 1, d_inner)` (cross-correlation, no flip)
//!    followed by `+ bias`, with the input left-padded by `kernel-1` zeros.
//! 2. [`selective_scan`] — the Mamba-1 selective state-space scan. One thread
//!    per `(batch, channel)`, sequential over time, hidden state `h[N]` held in
//!    a register array (`state_dim` is `comptime`). Recurrence per step:
//!    `da = exp(dt * A); h = h*da + dt*B*x; y = sum_n(h*C) + x*D`.
//!
//! Both kernels are `f32`-only (inference dtype). A naive sequential scan
//! underutilizes the GPU — a chunked/associative scan is the real perf fix and
//! is tracked as follow-up in the migration doc; correctness comes first here.

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Depthwise causal conv1d + bias.
///
/// Layouts (row-major, flattened `f32`):
/// - `input`  : `[B, C, L + K - 1]` — already left-padded with `K-1` zeros.
/// - `weight` : `[C, K]`            — depthwise conv weight (`[C,1,K]` flattened).
/// - `bias`   : `[C]`.
/// - `output` : `[B, C, L]`.
///
/// One thread per `(b, c)`; `ABSOLUTE_POS` linearises the `B*C` grid.
#[cube(launch)]
fn conv1d_depthwise(
    input: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    channels: u32,
    seq_len: u32,
    kernel: u32,
) {
    let channels = channels as usize;
    let seq_len = seq_len as usize;
    let kernel = kernel as usize;
    let idx = ABSOLUTE_POS;
    let total = output.len() / seq_len; // B*C
    if idx < total {
        let c = idx % channels;
        let padded_len = seq_len + kernel - 1;
        let in_base = idx * padded_len;
        let w_base = c * kernel;
        let out_base = idx * seq_len;
        let b_c = bias[c];
        for t in 0..seq_len {
            let mut acc = b_c;
            for j in 0..kernel {
                acc += input[in_base + t + j] * weight[w_base + j];
            }
            output[out_base + t] = acc;
        }
    }
}

/// Mamba-1 selective scan (stateless, `h` starts at zero).
///
/// Layouts (row-major, flattened `f32`):
/// - `delta` : `[B, L, C]` — `Δ` after softplus.
/// - `xs`    : `[B, L, C]` — post-conv, post-SiLU input `x`.
/// - `b_mat` : `[B, L, N]` — input-dependent `B`.
/// - `c_mat` : `[B, L, N]` — input-dependent `C`.
/// - `a`     : `[C, N]`    — `A = -exp(A_log)` (precomputed host-side).
/// - `d`     : `[C]`       — skip connection.
/// - `output`: `[B, L, C]` — `y`.
///
/// One thread per `(b, c)`. `state_dim` is `comptime` so `h[N]` lives in a
/// register array and the inner `n`-loops unroll.
#[cube(launch)]
fn selective_scan(
    delta: &Array<f32>,
    xs: &Array<f32>,
    b_mat: &Array<f32>,
    c_mat: &Array<f32>,
    a: &Array<f32>,
    d: &Array<f32>,
    output: &mut Array<f32>,
    channels: u32,
    seq_len: u32,
    #[comptime] state_dim: usize,
) {
    let channels = channels as usize;
    let seq_len = seq_len as usize;
    let idx = ABSOLUTE_POS;
    let total = xs.len() / seq_len; // B*C
    if idx < total {
        let b = idx / channels;
        let c = idx % channels;
        let d_c = d[c];
        let a_base = c * state_dim;

        // Hidden state h[N] in registers.
        let mut h = Array::<f32>::new(state_dim);
        for n in 0..state_dim {
            h[n] = 0.0f32;
        }

        for t in 0..seq_len {
            // (b, t, c) index into the [B, L, C] tensors.
            let bt_c = (b * seq_len + t) * channels + c;
            // (b, t, n) base into the [B, L, N] tensors.
            let bt_n = (b * seq_len + t) * state_dim;

            let dt = delta[bt_c];
            let x = xs[bt_c];

            let mut y = 0.0f32;
            for n in 0..state_dim {
                let da = (dt * a[a_base + n]).exp();
                let dbx = dt * b_mat[bt_n + n] * x;
                let hn = h[n] * da + dbx;
                h[n] = hn;
                y += hn * c_mat[bt_n + n];
            }
            output[bt_c] = y + x * d_c;
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke kernel (proves a runtime initialises + executes)
// ---------------------------------------------------------------------------

#[cube(launch)]
fn double_array(input: &Array<f32>, output: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        output[i] = input[i] * 2.0;
    }
}

/// Runs the trivial kernel on `R`, returning the doubled data. Used to check a
/// runtime can be created and dispatched to in this environment.
pub fn smoke_double<R: Runtime>(device: &R::Device, data: &[f32]) -> Vec<f32> {
    let client = R::client(device);
    let n = data.len();
    let input = client.create_from_slice(f32::as_bytes(data));
    let output = client.empty(std::mem::size_of_val(data));
    unsafe {
        double_array::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output.clone(), n),
        );
    }
    let bytes = client.read_one(output).unwrap();
    f32::from_bytes(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// Safe host wrappers (slice in / Vec out)
// ---------------------------------------------------------------------------

const THREADS_PER_CUBE: u32 = 64;

fn cube_count(total: u32) -> CubeCount {
    CubeCount::Static(total.div_ceil(THREADS_PER_CUBE), 1, 1)
}

/// Depthwise causal conv1d on runtime `R`.
///
/// `input` is `[batch, channels, seq_len + kernel - 1]` (already left-padded),
/// `weight` is `[channels, kernel]`, `bias` is `[channels]`. Returns
/// `[batch, channels, seq_len]`.
#[allow(clippy::too_many_arguments)]
pub fn run_conv1d<R: Runtime>(
    device: &R::Device,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), batch * channels * (seq_len + kernel - 1));
    assert_eq!(weight.len(), channels * kernel);
    assert_eq!(bias.len(), channels);

    let client = R::client(device);
    let out_len = batch * channels * seq_len;
    let input_h = client.create_from_slice(f32::as_bytes(input));
    let weight_h = client.create_from_slice(f32::as_bytes(weight));
    let bias_h = client.create_from_slice(f32::as_bytes(bias));
    let output_h = client.empty(out_len * std::mem::size_of::<f32>());

    let total = (batch * channels) as u32;
    unsafe {
        conv1d_depthwise::launch::<R>(
            &client,
            cube_count(total),
            CubeDim::new_1d(THREADS_PER_CUBE),
            ArrayArg::from_raw_parts(input_h, input.len()),
            ArrayArg::from_raw_parts(weight_h, weight.len()),
            ArrayArg::from_raw_parts(bias_h, bias.len()),
            ArrayArg::from_raw_parts(output_h.clone(), out_len),
            channels as u32,
            seq_len as u32,
            kernel as u32,
        );
    }
    let bytes = client.read_one(output_h).unwrap();
    f32::from_bytes(&bytes).to_vec()
}

/// Mamba-1 selective scan on runtime `R`.
///
/// `delta`, `xs` are `[batch, seq_len, channels]`; `b_mat`, `c_mat` are
/// `[batch, seq_len, state_dim]`; `a` is `[channels, state_dim]` (already
/// `-exp(A_log)`); `d` is `[channels]`. Returns `[batch, seq_len, channels]`.
#[allow(clippy::too_many_arguments)]
pub fn run_selective_scan<R: Runtime>(
    device: &R::Device,
    delta: &[f32],
    xs: &[f32],
    b_mat: &[f32],
    c_mat: &[f32],
    a: &[f32],
    d: &[f32],
    batch: usize,
    seq_len: usize,
    channels: usize,
    state_dim: usize,
) -> Vec<f32> {
    let n_tc = batch * seq_len * channels;
    let n_tn = batch * seq_len * state_dim;
    assert_eq!(delta.len(), n_tc);
    assert_eq!(xs.len(), n_tc);
    assert_eq!(b_mat.len(), n_tn);
    assert_eq!(c_mat.len(), n_tn);
    assert_eq!(a.len(), channels * state_dim);
    assert_eq!(d.len(), channels);

    let client = R::client(device);
    let delta_h = client.create_from_slice(f32::as_bytes(delta));
    let xs_h = client.create_from_slice(f32::as_bytes(xs));
    let b_h = client.create_from_slice(f32::as_bytes(b_mat));
    let c_h = client.create_from_slice(f32::as_bytes(c_mat));
    let a_h = client.create_from_slice(f32::as_bytes(a));
    let d_h = client.create_from_slice(f32::as_bytes(d));
    let output_h = client.empty(n_tc * std::mem::size_of::<f32>());

    let total = (batch * channels) as u32;
    unsafe {
        selective_scan::launch::<R>(
            &client,
            cube_count(total),
            CubeDim::new_1d(THREADS_PER_CUBE),
            ArrayArg::from_raw_parts(delta_h, delta.len()),
            ArrayArg::from_raw_parts(xs_h, xs.len()),
            ArrayArg::from_raw_parts(b_h, b_mat.len()),
            ArrayArg::from_raw_parts(c_h, c_mat.len()),
            ArrayArg::from_raw_parts(a_h, a.len()),
            ArrayArg::from_raw_parts(d_h, d.len()),
            ArrayArg::from_raw_parts(output_h.clone(), n_tc),
            channels as u32,
            seq_len as u32,
            state_dim,
        );
    }
    let bytes = client.read_one(output_h).unwrap();
    f32::from_bytes(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// CPU scalar reference — identical math, used as the parity baseline
// ---------------------------------------------------------------------------

/// Scalar reference for [`run_conv1d`] (same layouts).
#[allow(clippy::too_many_arguments)]
pub fn reference_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel: usize,
) -> Vec<f32> {
    let padded_len = seq_len + kernel - 1;
    let mut out = vec![0.0f32; batch * channels * seq_len];
    for bc in 0..batch * channels {
        let c = bc % channels;
        let in_base = bc * padded_len;
        let w_base = c * kernel;
        let out_base = bc * seq_len;
        for t in 0..seq_len {
            let mut acc = bias[c];
            for j in 0..kernel {
                acc += input[in_base + t + j] * weight[w_base + j];
            }
            out[out_base + t] = acc;
        }
    }
    out
}

/// Scalar reference for [`run_selective_scan`] (same layouts).
#[allow(clippy::too_many_arguments)]
pub fn reference_selective_scan(
    delta: &[f32],
    xs: &[f32],
    b_mat: &[f32],
    c_mat: &[f32],
    a: &[f32],
    d: &[f32],
    batch: usize,
    seq_len: usize,
    channels: usize,
    state_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_len * channels];
    let mut h = vec![0.0f32; state_dim];
    for b in 0..batch {
        // `c` indexes several arrays and feeds index math — a genuine index loop.
        #[allow(clippy::needless_range_loop)]
        for c in 0..channels {
            for hv in h.iter_mut() {
                *hv = 0.0;
            }
            let a_base = c * state_dim;
            for t in 0..seq_len {
                let bt_c = (b * seq_len + t) * channels + c;
                let bt_n = (b * seq_len + t) * state_dim;
                let dt = delta[bt_c];
                let x = xs[bt_c];
                let mut y = 0.0f32;
                for n in 0..state_dim {
                    let da = (dt * a[a_base + n]).exp();
                    let dbx = dt * b_mat[bt_n + n] * x;
                    h[n] = h[n] * da + dbx;
                    y += h[n] * c_mat[bt_n + n];
                }
                out[bt_c] = y + x * d[c];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    type Rt = cubecl::wgpu::WgpuRuntime;

    fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let max_abs = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            / max_abs
    }

    // Deterministic pseudo-random fill in [-1, 1].
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn wgpu_smoke() {
        let out = smoke_double::<Rt>(&Default::default(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn conv1d_matches_reference() {
        let (batch, channels, seq_len, kernel) = (2usize, 5usize, 7usize, 4usize);
        let padded_len = seq_len + kernel - 1;
        let input = fill(batch * channels * padded_len, 1);
        let weight = fill(channels * kernel, 2);
        let bias = fill(channels, 3);

        let got = run_conv1d::<Rt>(
            &Default::default(),
            &input,
            &weight,
            &bias,
            batch,
            channels,
            seq_len,
            kernel,
        );
        let want = reference_conv1d(&input, &weight, &bias, batch, channels, seq_len, kernel);
        let diff = max_rel_diff(&got, &want);
        assert!(diff < 1e-4, "conv1d parity diff {diff}");
    }

    // Parity against the *actual* Candle ops used by `model::MambaMixer`
    // (not just our scalar reference) — this is the real migration guarantee.
    #[test]
    fn conv1d_matches_candle() {
        use candle_core::{Device, Tensor};
        let (batch, channels, seq_len, kernel) = (2usize, 4usize, 9usize, 4usize);
        let dev = Device::Cpu;

        let x_flat = fill(batch * channels * seq_len, 20);
        let w_flat = fill(channels * kernel, 21);
        let bias = fill(channels, 22);

        // Candle reference: left-pad k-1, depthwise conv1d, add bias.
        let x = Tensor::from_vec(x_flat.clone(), (batch, channels, seq_len), &dev).unwrap();
        let weight = Tensor::from_vec(w_flat.clone(), (channels, 1, kernel), &dev).unwrap();
        let bias_t = Tensor::from_vec(bias.clone(), channels, &dev).unwrap();
        let padded = x.pad_with_zeros(2, kernel - 1, 0).unwrap();
        let conv = padded.conv1d(&weight, 0, 1, 1, channels).unwrap();
        let conv = conv.narrow(2, 0, seq_len).unwrap();
        let conv = conv
            .broadcast_add(&bias_t.reshape((1, channels, 1)).unwrap())
            .unwrap();
        let want: Vec<f32> = conv.flatten_all().unwrap().to_vec1().unwrap();

        // CubeCL: feed the already-padded input.
        let padded_flat: Vec<f32> = padded.flatten_all().unwrap().to_vec1().unwrap();
        let got = run_conv1d::<Rt>(
            &Default::default(),
            &padded_flat,
            &w_flat,
            &bias,
            batch,
            channels,
            seq_len,
            kernel,
        );
        let diff = max_rel_diff(&got, &want);
        assert!(diff < 1e-4, "conv1d vs Candle diff {diff}");
    }

    #[test]
    fn selective_scan_matches_candle() {
        use candle_core::{Device, Tensor};
        let (batch, seq_len, channels, state_dim) = (2usize, 5usize, 4usize, 16usize);
        let dev = Device::Cpu;

        let delta: Vec<f32> = fill(batch * seq_len * channels, 30)
            .iter()
            .map(|v| 0.3 * (v + 1.5))
            .collect();
        let xs = fill(batch * seq_len * channels, 31);
        let b_mat = fill(batch * seq_len * state_dim, 32);
        let c_mat = fill(batch * seq_len * state_dim, 33);
        let a: Vec<f32> = fill(channels * state_dim, 34)
            .iter()
            .map(|v| -v.exp())
            .collect();
        let d = fill(channels, 35);

        // Candle reference mirroring model::MambaMixer's sequential scan
        // (stateless, h starts at zero), operating on [B, L, C]/[B, L, N].
        let delta_t = Tensor::from_vec(delta.clone(), (batch, seq_len, channels), &dev).unwrap();
        let xs_t = Tensor::from_vec(xs.clone(), (batch, seq_len, channels), &dev).unwrap();
        let b_t = Tensor::from_vec(b_mat.clone(), (batch, seq_len, state_dim), &dev).unwrap();
        let c_t = Tensor::from_vec(c_mat.clone(), (batch, seq_len, state_dim), &dev).unwrap();
        let a_t = Tensor::from_vec(a.clone(), (channels, state_dim), &dev).unwrap();
        let d_t = Tensor::from_vec(d.clone(), channels, &dev).unwrap();

        let mut h =
            Tensor::zeros((batch, channels, state_dim), candle_core::DType::F32, &dev).unwrap();
        let mut ys = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let dt = delta_t.narrow(1, t, 1).unwrap().squeeze(1).unwrap(); // [B, C]
            let xt = xs_t.narrow(1, t, 1).unwrap().squeeze(1).unwrap(); // [B, C]
            let bt = b_t.narrow(1, t, 1).unwrap().squeeze(1).unwrap(); // [B, N]
            let ct = c_t.narrow(1, t, 1).unwrap().squeeze(1).unwrap(); // [B, N]
            let dt_e = dt.unsqueeze(2).unwrap(); // [B, C, 1]
            let da = dt_e
                .broadcast_mul(&a_t.unsqueeze(0).unwrap())
                .unwrap()
                .exp()
                .unwrap();
            let dbx = dt_e
                .broadcast_mul(&bt.unsqueeze(1).unwrap())
                .unwrap()
                .broadcast_mul(&xt.unsqueeze(2).unwrap())
                .unwrap();
            h = (h.mul(&da).unwrap() + dbx).unwrap();
            let yt = h
                .broadcast_mul(&ct.unsqueeze(1).unwrap())
                .unwrap()
                .sum(2)
                .unwrap();
            let yt = (yt + xt.broadcast_mul(&d_t.unsqueeze(0).unwrap()).unwrap()).unwrap();
            ys.push(yt.unsqueeze(1).unwrap());
        }
        let y = Tensor::cat(&ys, 1).unwrap(); // [B, L, C]
        let want: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();

        let got = run_selective_scan::<Rt>(
            &Default::default(),
            &delta,
            &xs,
            &b_mat,
            &c_mat,
            &a,
            &d,
            batch,
            seq_len,
            channels,
            state_dim,
        );
        let diff = max_rel_diff(&got, &want);
        assert!(diff < 1e-4, "selective_scan vs Candle diff {diff}");
    }

    #[test]
    fn selective_scan_matches_reference() {
        let (batch, seq_len, channels, state_dim) = (2usize, 6usize, 8usize, 16usize);
        let delta = fill(batch * seq_len * channels, 10)
            .iter()
            .map(|v| 0.5 * (v + 1.0)) // Δ > 0, as after softplus
            .collect::<Vec<_>>();
        let xs = fill(batch * seq_len * channels, 11);
        let b_mat = fill(batch * seq_len * state_dim, 12);
        let c_mat = fill(batch * seq_len * state_dim, 13);
        // A = -exp(A_log) < 0
        let a = fill(channels * state_dim, 14)
            .iter()
            .map(|v| -v.exp())
            .collect::<Vec<_>>();
        let d = fill(channels, 15);

        let got = run_selective_scan::<Rt>(
            &Default::default(),
            &delta,
            &xs,
            &b_mat,
            &c_mat,
            &a,
            &d,
            batch,
            seq_len,
            channels,
            state_dim,
        );
        let want = reference_selective_scan(
            &delta, &xs, &b_mat, &c_mat, &a, &d, batch, seq_len, channels, state_dim,
        );
        let diff = max_rel_diff(&got, &want);
        assert!(diff < 1e-4, "selective_scan parity diff {diff}");
    }
}
