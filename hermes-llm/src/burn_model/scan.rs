//! Backend dispatch for the Mamba selective scan.
//!
//! ndarray uses the readable tensor-op reference. Burn's GPU backends expose
//! their resident `CubeTensor` handles, allowing the same CubeCL kernel to run
//! on Metal and CUDA without staging through host memory.

use burn::prelude::*;

/// Backend capability required by the Burn Mamba mixer.
pub trait MambaBackend: Backend {
    /// Returns `(y, final_h)` for the Mamba recurrence.
    #[allow(clippy::too_many_arguments)]
    fn selective_scan(
        delta: Tensor<Self, 3>,
        xs: Tensor<Self, 3>,
        b_mat: Tensor<Self, 3>,
        c_mat: Tensor<Self, 3>,
        a: Tensor<Self, 2>,
        d: Tensor<Self, 1>,
        h: Tensor<Self, 3>,
        state_dim: usize,
    ) -> (Tensor<Self, 3>, Tensor<Self, 3>);
}

#[allow(clippy::too_many_arguments)]
fn reference_selective_scan<B: Backend>(
    delta: Tensor<B, 3>,
    xs: Tensor<B, 3>,
    b_mat: Tensor<B, 3>,
    c_mat: Tensor<B, 3>,
    a: Tensor<B, 2>,
    d: Tensor<B, 1>,
    mut h: Tensor<B, 3>,
    state_dim: usize,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [batch, seq_len, channels] = xs.dims();
    assert_eq!(state_dim, h.dims()[2]);
    let mut ys = Vec::with_capacity(seq_len);
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
    }
    (Tensor::cat(ys, 1), h)
}

impl MambaBackend for burn_ndarray::NdArray {
    fn selective_scan(
        delta: Tensor<Self, 3>,
        xs: Tensor<Self, 3>,
        b_mat: Tensor<Self, 3>,
        c_mat: Tensor<Self, 3>,
        a: Tensor<Self, 2>,
        d: Tensor<Self, 1>,
        h: Tensor<Self, 3>,
        state_dim: usize,
    ) -> (Tensor<Self, 3>, Tensor<Self, 3>) {
        reference_selective_scan(delta, xs, b_mat, c_mat, a, d, h, state_dim)
    }
}

impl<B: Backend> MambaBackend for burn_autodiff::Autodiff<B> {
    fn selective_scan(
        delta: Tensor<Self, 3>,
        xs: Tensor<Self, 3>,
        b_mat: Tensor<Self, 3>,
        c_mat: Tensor<Self, 3>,
        a: Tensor<Self, 2>,
        d: Tensor<Self, 1>,
        h: Tensor<Self, 3>,
        state_dim: usize,
    ) -> (Tensor<Self, 3>, Tensor<Self, 3>) {
        reference_selective_scan(delta, xs, b_mat, c_mat, a, d, h, state_dim)
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use burn::tensor::{Shape, Tensor as BurnTensor, TensorPrimitive};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::element::{BoolElement, IntElement};
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};

    use super::MambaBackend;

    const THREADS_PER_CUBE: u32 = 64;

    #[cube(launch)]
    fn selective_scan_stateful(
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
        seq_len: u32,
        #[comptime] state_dim: usize,
    ) {
        let channels = channels as usize;
        let seq_len = seq_len as usize;
        let idx = ABSOLUTE_POS;
        let total = xs.len() / seq_len;
        if idx < total {
            let batch = idx / channels;
            let channel = idx % channels;
            let state_base = idx * state_dim;
            let a_base = channel * state_dim;
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
                    let da = (dt * a[a_base + n]).exp();
                    let next = state[n] * da + dt * b_mat[btn + n] * x;
                    state[n] = next;
                    out += next * c_mat[btn + n];
                }
                y[btc] = out + x * d[channel];
            }
            for n in 0..state_dim {
                h_out[state_base + n] = state[n];
            }
        }
    }

    fn contiguous<R: CubeRuntime, const D: usize, I: IntElement, BT: BoolElement>(
        tensor: BurnTensor<CubeBackend<R, f32, I, BT>, D>,
    ) -> CubeTensor<R> {
        burn_cubecl::kernel::into_contiguous(tensor.into_primitive().tensor())
    }

    impl<R, I, BT> MambaBackend for CubeBackend<R, f32, I, BT>
    where
        R: CubeRuntime,
        I: IntElement,
        BT: BoolElement,
    {
        fn selective_scan(
            delta: BurnTensor<Self, 3>,
            xs: BurnTensor<Self, 3>,
            b_mat: BurnTensor<Self, 3>,
            c_mat: BurnTensor<Self, 3>,
            a: BurnTensor<Self, 2>,
            d: BurnTensor<Self, 1>,
            h: BurnTensor<Self, 3>,
            state_dim: usize,
        ) -> (BurnTensor<Self, 3>, BurnTensor<Self, 3>) {
            let [batch, seq_len, channels] = xs.dims();
            let delta = contiguous(delta);
            let xs = contiguous(xs);
            let b_mat = contiguous(b_mat);
            let c_mat = contiguous(c_mat);
            let a = contiguous(a);
            let d = contiguous(d);
            let h = contiguous(h);
            let client = xs.client.clone();
            let device = xs.device.clone();
            let dtype = xs.dtype;
            let y = burn_cubecl::ops::numeric::empty_device_contiguous_dtype(
                client.clone(),
                device.clone(),
                Shape::new([batch, seq_len, channels]),
                dtype,
            );
            let h_out = burn_cubecl::ops::numeric::empty_device_contiguous_dtype(
                client.clone(),
                device,
                Shape::new([batch, channels, state_dim]),
                dtype,
            );

            let total = (batch * channels) as u32;
            selective_scan_stateful::launch::<R>(
                &client,
                CubeCount::Static(total.div_ceil(THREADS_PER_CUBE), 1, 1),
                CubeDim::new_1d(THREADS_PER_CUBE),
                delta.into_array_arg(),
                xs.into_array_arg(),
                b_mat.into_array_arg(),
                c_mat.into_array_arg(),
                a.into_array_arg(),
                d.into_array_arg(),
                h.into_array_arg(),
                y.clone().into_array_arg(),
                h_out.clone().into_array_arg(),
                channels as u32,
                seq_len as u32,
                state_dim,
            );

            (
                BurnTensor::from_primitive(TensorPrimitive::Float(y)),
                BurnTensor::from_primitive(TensorPrimitive::Float(h_out)),
            )
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::NdArray;
    use burn_wgpu::Wgpu;

    use super::MambaBackend;

    fn values(len: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * scale).sin() * 0.25 + offset)
            .collect()
    }

    #[test]
    fn test_cubecl_selective_scan_matches_ndarray_reference() {
        type Cpu = NdArray;
        type Gpu = Wgpu;
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

        let (cpu_y, cpu_h) = Cpu::selective_scan(
            tensor!(Cpu, &cpu, delta, [batch, seq_len, channels]),
            tensor!(Cpu, &cpu, xs, [batch, seq_len, channels]),
            tensor!(Cpu, &cpu, b_mat, [batch, seq_len, state_dim]),
            tensor!(Cpu, &cpu, c_mat, [batch, seq_len, state_dim]),
            tensor!(Cpu, &cpu, a, [channels, state_dim]),
            tensor!(Cpu, &cpu, d, [channels]),
            tensor!(Cpu, &cpu, h, [batch, channels, state_dim]),
            state_dim,
        );
        let (gpu_y, gpu_h) = Gpu::selective_scan(
            tensor!(Gpu, &gpu, delta, [batch, seq_len, channels]),
            tensor!(Gpu, &gpu, xs, [batch, seq_len, channels]),
            tensor!(Gpu, &gpu, b_mat, [batch, seq_len, state_dim]),
            tensor!(Gpu, &gpu, c_mat, [batch, seq_len, state_dim]),
            tensor!(Gpu, &gpu, a, [channels, state_dim]),
            tensor!(Gpu, &gpu, d, [channels]),
            tensor!(Gpu, &gpu, h, [batch, channels, state_dim]),
            state_dim,
        );

        let max_diff = |cpu: TensorData, gpu: TensorData| {
            cpu.to_vec::<f32>()
                .unwrap()
                .into_iter()
                .zip(gpu.to_vec::<f32>().unwrap())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max)
        };
        assert!(max_diff(cpu_y.into_data(), gpu_y.into_data()) < 1e-5);
        assert!(max_diff(cpu_h.into_data(), gpu_h.into_data()) < 1e-5);
    }
}
