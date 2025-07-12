//! Fused selective-scan custom op for Mamba inference.
//!
//! Design: docs/metal-selective-scan.md. The per-timestep Candle-op loop in
//! `MambaMixer::forward_with_state` is pathological on Metal (one dispatch per
//! tiny op, plus CPU fallbacks that force GPU↔CPU round-trips every scan step).
//! This op runs the whole scan — all timesteps, state in registers — in a
//! single kernel dispatch. The CPU implementation is the required scalar
//! fallback and is also faster than the tensor-op loop, so CPU serving uses it
//! too. CUDA keeps the reference loop (no fused CUDA inference kernel yet).
//!
//! Candle's `CustomOp3` allows 3 tensor inputs / 1 output, so inputs and
//! outputs are packed (all f32, contiguous):
//!   in1: concat(xs, Δ)  last-dim          [B, L, 2·di]
//!   in2: concat(Bmat, Cmat) last-dim      [B, L, 2·N]
//!   in3: concat(A.flat, D, h0.flat) 1-D   [di·N + di + B·di·N]
//!   out: concat(y.flat, h_final.flat) 1-D [B·L·di + B·di·N]

use candle_core::{CpuStorage, CustomOp3, Device, Layout, Result, Shape};

/// SSM state size the kernel supports: h lives in a fixed-size per-thread
/// register array in the Metal shader.
pub const MAX_STATE_DIM: usize = 64;

pub struct FusedSelectiveScan {
    pub batch: usize,
    pub seq_len: usize,
    pub d_inner: usize,
    pub state_dim: usize,
}

impl FusedSelectiveScan {
    fn out_len(&self) -> usize {
        self.batch * self.seq_len * self.d_inner + self.batch * self.d_inner * self.state_dim
    }
}

/// Whether the fused scan should be used for this device/dtype/state size.
/// CUDA returns false: the reference loop there is made of native kernels and
/// we don't ship a fused CUDA inference kernel yet.
pub fn fused_supported(device: &Device, dtype: candle_core::DType, state_dim: usize) -> bool {
    if dtype != candle_core::DType::F32 {
        return false;
    }
    if state_dim > MAX_STATE_DIM {
        warn_once(
            "ssm state_dim exceeds fused-scan register budget; using per-step scan (slow on Metal)",
        );
        return false;
    }
    match device {
        Device::Cpu => true,
        #[cfg(feature = "metal")]
        Device::Metal(m) => metal::available(m),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

fn warn_once(msg: &str) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| tracing::warn!("{msg}"));
}

fn contiguous<'a>(s: &'a CpuStorage, l: &Layout, name: &str) -> Result<&'a [f32]> {
    let (start, end) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg(format!("fused scan: {name} not contiguous")))?;
    Ok(&s.as_slice::<f32>()?[start..end])
}

impl CustomOp3 for FusedSelectiveScan {
    fn name(&self) -> &'static str {
        "fused_selective_scan"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, l, di, n) = (self.batch, self.seq_len, self.d_inner, self.state_dim);
        let xd = contiguous(s1, l1, "xs|delta")?;
        let bc = contiguous(s2, l2, "B|C")?;
        let adh = contiguous(s3, l3, "A|D|h0")?;

        let mut out = vec![0f32; self.out_len()];
        let h0_base = di * n + di;
        for bi in 0..b {
            for c in 0..di {
                let a_row = &adh[c * n..(c + 1) * n];
                let d_c = adh[di * n + c];
                let mut h: [f32; MAX_STATE_DIM] = [0.0; MAX_STATE_DIM];
                h[..n].copy_from_slice(&adh[h0_base + (bi * di + c) * n..][..n]);
                for t in 0..l {
                    let row = (bi * l + t) * 2 * di;
                    let x = xd[row + c];
                    let dt = xd[row + di + c];
                    let brow = &bc[(bi * l + t) * 2 * n..];
                    let mut y = 0f32;
                    for (j, (h_j, a_j)) in h[..n].iter_mut().zip(a_row).enumerate() {
                        let da = (dt * a_j).exp();
                        *h_j = *h_j * da + dt * brow[j] * x;
                        y += *h_j * brow[n + j];
                    }
                    y += d_c * x;
                    out[(bi * l + t) * di + c] = y;
                }
                let ho = b * l * di + (bi * di + c) * n;
                out[ho..ho + n].copy_from_slice(&h[..n]);
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(self.out_len())))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let params = [
            self.batch as u32,
            self.seq_len as u32,
            self.d_inner as u32,
            self.state_dim as u32,
        ];
        metal::fwd(
            metal::Kernel::Scan,
            params,
            self.batch * self.d_inner,
            self.out_len(),
            Shape::from(self.out_len()),
            [(s1, l1, "xs|delta"), (s2, l2, "B|C"), (s3, l3, "A|D|h0")],
            s1.device(),
        )
    }
}

/// Depthwise (groups == channels) causal conv1d with the bias folded in.
///
/// Candle lowers grouped conv1d to one conv per group; with
/// groups == d_inner that is ~16k dispatches per decoded token across the
/// Mamba layers — measured at 98% of Metal decode CPU time. One dispatch
/// (and on CPU, one tight loop) instead. Inputs (f32, contiguous):
///   in1: padded input [B, di, L_in]
///   in2: weight, flattened [di · K]
///   in3: bias [di]
///   out: [B, di, L_in − K + 1]
pub struct FusedDepthwiseConv1d {
    pub batch: usize,
    pub d_inner: usize,
    pub l_in: usize,
    pub kernel: usize,
}

impl FusedDepthwiseConv1d {
    fn l_out(&self) -> usize {
        self.l_in - self.kernel + 1
    }

    fn out_shape(&self) -> Shape {
        Shape::from((self.batch, self.d_inner, self.l_out()))
    }
}

impl CustomOp3 for FusedDepthwiseConv1d {
    fn name(&self) -> &'static str {
        "fused_depthwise_conv1d"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, di, l_in, k) = (self.batch, self.d_inner, self.l_in, self.kernel);
        let input = contiguous(s1, l1, "conv input")?;
        let w = contiguous(s2, l2, "conv weight")?;
        let bias = contiguous(s3, l3, "conv bias")?;

        let l_out = self.l_out();
        let mut out = vec![0f32; b * di * l_out];
        for bi in 0..b {
            for c in 0..di {
                let row = &input[(bi * di + c) * l_in..][..l_in];
                let wc = &w[c * k..(c + 1) * k];
                let orow = &mut out[(bi * di + c) * l_out..][..l_out];
                for (t, o) in orow.iter_mut().enumerate() {
                    let mut acc = bias[c];
                    for (j, wj) in wc.iter().enumerate() {
                        acc += wj * row[t + j];
                    }
                    *o = acc;
                }
            }
        }
        Ok((CpuStorage::F32(out), self.out_shape()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let params = [
            self.batch as u32,
            self.d_inner as u32,
            self.l_in as u32,
            self.kernel as u32,
        ];
        metal::fwd(
            metal::Kernel::Conv,
            params,
            self.batch * self.d_inner,
            self.batch * self.d_inner * self.l_out(),
            self.out_shape(),
            [
                (s1, l1, "conv input"),
                (s2, l2, "conv weight"),
                (s3, l3, "conv bias"),
            ],
            s1.device(),
        )
    }
}

#[cfg(feature = "metal")]
mod metal {
    use candle_core::{DType, Layout, MetalDevice, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
    use objc2_metal::MTLSize;
    use std::sync::OnceLock;

    const MSL: &str = include_str!("selective_scan.metal");

    #[derive(Clone, Copy)]
    pub(super) enum Kernel {
        Scan,
        Conv,
    }

    pub(super) struct Pipelines {
        scan: ComputePipeline,
        conv: ComputePipeline,
    }

    /// One pipeline set per process. Macs have a single GPU; if we ever serve
    /// on multi-GPU Metal this needs a per-device cache.
    static PIPELINES: OnceLock<Option<Pipelines>> = OnceLock::new();

    fn pipeline(device: &MetalDevice) -> Option<&'static Pipelines> {
        PIPELINES
            .get_or_init(|| match build(device) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(
                        "fused Metal selective-scan kernels failed to build; \
                         falling back to per-step ops (very slow on Metal): {e}"
                    );
                    None
                }
            })
            .as_ref()
    }

    pub(super) fn available(device: &MetalDevice) -> bool {
        pipeline(device).is_some()
    }

    fn build(device: &MetalDevice) -> Result<Pipelines> {
        let raw = device.metal_device();
        let lib = raw
            .new_library_with_source(MSL, None)
            .map_err(|e| candle_core::Error::Msg(format!("compile: {e}")))?;
        let make = |name: &str| -> Result<ComputePipeline> {
            let func = lib
                .get_function(name, None)
                .map_err(|e| candle_core::Error::Msg(format!("get_function {name}: {e}")))?;
            raw.new_compute_pipeline_state_with_function(&func)
                .map_err(|e| candle_core::Error::Msg(format!("pipeline {name}: {e}")))
        };
        Ok(Pipelines {
            scan: make("fused_selective_scan_f32")?,
            conv: make("depthwise_conv1d_f32")?,
        })
    }

    /// Encode one fused-kernel dispatch: 4 u32 params at buffer 0, three
    /// contiguous f32 inputs at 1..=3, one f32 output at 4, one thread per
    /// (batch, channel).
    pub(super) fn fwd(
        which: Kernel,
        params: [u32; 4],
        threads: usize,
        out_len: usize,
        out_shape: Shape,
        inputs: [(&MetalStorage, &Layout, &str); 3],
        device: &MetalDevice,
    ) -> Result<(MetalStorage, Shape)> {
        for (_, l, name) in inputs {
            if l.contiguous_offsets().is_none() {
                candle_core::bail!("fused kernel: {name} not contiguous");
            }
        }
        let pipelines = pipeline(device).ok_or_else(|| {
            candle_core::Error::Msg("fused kernel pipelines unavailable".to_string())
        })?;
        let pipeline = match which {
            Kernel::Scan => &pipelines.scan,
            Kernel::Conv => &pipelines.conv,
        };

        let out_buf = device
            .new_buffer_builder()
            .with_size_for(out_len, DType::F32)
            .with_label("fused_kernel")
            .build()?;

        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(0, &params);
        let f32sz = DType::F32.size_in_bytes();
        for (i, (s, l, _)) in inputs.iter().enumerate() {
            encoder.set_input_buffer(i + 1, Some(s.buffer()), l.start_offset() * f32sz);
        }
        encoder.set_output_buffer(4, Some(&out_buf), 0);

        let tg = pipeline
            .max_total_threads_per_threadgroup()
            .min(threads)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        drop(guard);

        Ok((
            MetalStorage::new(out_buf, device.clone(), out_len, DType::F32),
            out_shape,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    /// Independent reference: the same recurrence written directly from the
    /// math (h ← h·exp(ΔA) + ΔBx; y = Σ h·C + Dx), used to pin both backends.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        xs: &[f32],
        delta: &[f32],
        bmat: &[f32],
        cmat: &[f32],
        a: &[f32],
        d: &[f32],
        h0: &[f32],
        (b, l, di, n): (usize, usize, usize, usize),
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0f32; b * l * di];
        let mut hf = h0.to_vec();
        for bi in 0..b {
            for c in 0..di {
                for t in 0..l {
                    let x = xs[(bi * l + t) * di + c];
                    let dt = delta[(bi * l + t) * di + c];
                    let mut acc = 0f32;
                    for j in 0..n {
                        let h = &mut hf[(bi * di + c) * n + j];
                        *h = *h * (dt * a[c * n + j]).exp() + dt * bmat[(bi * l + t) * n + j] * x;
                        acc += *h * cmat[(bi * l + t) * n + j];
                    }
                    y[(bi * l + t) * di + c] = acc + d[c] * x;
                }
            }
        }
        (y, hf)
    }

    fn rng_fill(v: &mut [f32], seed: &mut u64) {
        for x in v {
            // xorshift; values in [-1, 1)
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *x = ((*seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        }
    }

    fn run_case(device: &Device, dims: (usize, usize, usize, usize)) -> Result<()> {
        let (b, l, di, n) = dims;
        let mut seed = 0x9E3779B97F4A7C15;
        let mut xs = vec![0f32; b * l * di];
        let mut delta = vec![0f32; b * l * di];
        let mut bmat = vec![0f32; b * l * n];
        let mut cmat = vec![0f32; b * l * n];
        let mut a = vec![0f32; di * n];
        let mut d = vec![0f32; di];
        let mut h0 = vec![0f32; b * di * n];
        for v in [
            &mut xs, &mut delta, &mut bmat, &mut cmat, &mut a, &mut d, &mut h0,
        ] {
            rng_fill(v, &mut seed);
        }
        // Δ must be positive (softplus output) and A negative (−exp(a_log))
        for v in &mut delta {
            *v = v.abs() + 0.01;
        }
        for v in &mut a {
            *v = -v.abs() - 0.01;
        }

        let (want_y, want_h) = reference(&xs, &delta, &bmat, &cmat, &a, &d, &h0, dims);

        let t_xs = Tensor::from_slice(&xs, (b, l, di), device)?;
        let t_dt = Tensor::from_slice(&delta, (b, l, di), device)?;
        let t_b = Tensor::from_slice(&bmat, (b, l, n), device)?;
        let t_c = Tensor::from_slice(&cmat, (b, l, n), device)?;
        let t_a = Tensor::from_slice(&a, di * n, device)?;
        let t_d = Tensor::from_slice(&d, di, device)?;
        let t_h = Tensor::from_slice(&h0, b * di * n, device)?;

        let xd = Tensor::cat(&[&t_xs, &t_dt], 2)?.contiguous()?;
        let bc = Tensor::cat(&[&t_b, &t_c], 2)?.contiguous()?;
        let adh = Tensor::cat(&[&t_a, &t_d, &t_h], 0)?.contiguous()?;
        let op = FusedSelectiveScan {
            batch: b,
            seq_len: l,
            d_inner: di,
            state_dim: n,
        };
        let out = xd.apply_op3_no_bwd(&bc, &adh, &op)?.to_vec1::<f32>()?;

        let (got_y, got_h) = out.split_at(b * l * di);
        for (i, (g, w)) in got_y.iter().zip(&want_y).enumerate() {
            assert!((g - w).abs() < 1e-4, "y[{i}]: got {g}, want {w}");
        }
        for (i, (g, w)) in got_h.iter().zip(&want_h).enumerate() {
            assert!((g - w).abs() < 1e-4, "h[{i}]: got {g}, want {w}");
        }
        Ok(())
    }

    /// Pin the fused depthwise conv against candle's own grouped conv1d.
    fn run_conv_case(device: &Device, (b, di, l, k): (usize, usize, usize, usize)) -> Result<()> {
        let mut seed = 0xDEADBEEFCAFED00D;
        let l_in = l + k - 1;
        let mut input = vec![0f32; b * di * l_in];
        let mut w = vec![0f32; di * k];
        let mut bias = vec![0f32; di];
        for v in [&mut input, &mut w, &mut bias] {
            rng_fill(v, &mut seed);
        }
        let t_in = Tensor::from_slice(&input, (b, di, l_in), device)?;
        let t_w3 = Tensor::from_slice(&w, (di, 1, k), device)?;
        let t_w = Tensor::from_slice(&w, di * k, device)?;
        let t_bias = Tensor::from_slice(&bias, di, device)?;

        let want = t_in
            .conv1d(&t_w3, 0, 1, 1, di)?
            .broadcast_add(&t_bias.reshape((1, di, 1))?)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let op = FusedDepthwiseConv1d {
            batch: b,
            d_inner: di,
            l_in,
            kernel: k,
        };
        let got = t_in
            .apply_op3_no_bwd(&t_w, &t_bias, &op)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-4, "conv[{i}]: got {g}, want {w}");
        }
        Ok(())
    }

    #[test]
    fn fused_conv_cpu_matches_candle() {
        run_conv_case(&Device::Cpu, (2, 8, 1, 4)).unwrap(); // decode
        run_conv_case(&Device::Cpu, (2, 8, 9, 4)).unwrap(); // prefill
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_conv_metal_matches_candle() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipping");
            return;
        };
        run_conv_case(&device, (2, 8, 1, 4)).unwrap();
        run_conv_case(&device, (2, 8, 9, 4)).unwrap();
    }

    #[test]
    fn fused_cpu_matches_reference_decode() {
        // L=1 is the decode hot path
        run_case(&Device::Cpu, (2, 1, 8, 16)).unwrap();
    }

    #[test]
    fn fused_cpu_matches_reference_prefill() {
        run_case(&Device::Cpu, (2, 9, 8, 16)).unwrap();
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_metal_matches_reference() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("no Metal device; skipping");
            return;
        };
        run_case(&device, (2, 1, 8, 16)).unwrap();
        run_case(&device, (2, 9, 8, 16)).unwrap();
        // register-budget edge: state_dim == MAX_STATE_DIM
        run_case(&device, (1, 3, 4, MAX_STATE_DIM)).unwrap();
    }
}
