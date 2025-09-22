//! CPU-vs-GPU parity for the selective scan.
//!
//! Every geometry runs the same data through the differentiable reference
//! on the CPU and the fused kernels on the GPU device the build enables,
//! comparing outputs and all input gradients. Shapes are chosen
//! adversarially: partial tail segments, channel tiles with overhang, the
//! supported state widths, and the loud refusal of unsupported ones.

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
    assert_cubecl_selective_scan_backward_geometry(batch, 37, 3, 4);
}

fn assert_cubecl_selective_scan_backward_geometry(
    batch: usize,
    seq_len: usize,
    channels: usize,
    state_dim: usize,
) {
    let cpu = Device::ndarray().autodiff();
    let gpu = gpu_device().autodiff();
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
            let delta = Tensor::<3>::from_data(TensorData::new(data.0.clone(), shapes.0), $device)
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
fn test_cubecl_selective_scan_small_batch_backward_matches_ndarray() {
    assert_cubecl_selective_scan_backward(2);
}

/// BF16 sequence tensors through the checkpointed training path against
/// the f32 CPU reference over the same bf16-quantized data: differences
/// are bounded by the BF16 output/gradient quantization, not kernel math.
#[test]
fn test_cubecl_selective_scan_bf16_io_matches_f32_reference() {
    use burn::tensor::FloatDType;

    fn quantize(values: Vec<f32>) -> Vec<f32> {
        values
            .into_iter()
            .map(|value| half::bf16::from_f32(value).to_f32())
            .collect()
    }

    let cpu = Device::ndarray().autodiff();
    let gpu = gpu_device().autodiff();
    let (batch, seq_len, channels, state_dim) = (3, 37, 3, 4);
    let shapes = (
        [batch, seq_len, channels],
        [batch, seq_len, state_dim],
        [channels, state_dim],
        [channels],
        [batch, channels, state_dim],
    );
    // Strictly negative state matrix keeps the recurrence bounded, so the
    // comparison measures kernel arithmetic rather than the BF16
    // quantization of a geometrically growing output.
    let a_data: Vec<f32> = values(channels * state_dim, 0.11, -0.4)
        .into_iter()
        .map(|value| -value.abs() - 0.05)
        .collect();
    let data = (
        quantize(values(batch * seq_len * channels, 0.13, 0.08)),
        quantize(values(batch * seq_len * channels, 0.17, -0.03)),
        quantize(values(batch * seq_len * state_dim, 0.19, 0.02)),
        quantize(values(batch * seq_len * state_dim, 0.23, -0.01)),
        a_data,
        values(channels, 0.07, 0.9),
        values(batch * channels * state_dim, 0.05, 0.01),
    );
    let weight_data = quantize(values(batch * seq_len * channels, 0.29, 0.5));

    macro_rules! run {
        ($device:expr, $bf16:expr) => {{
            let cast = |tensor: Tensor<3>| {
                if $bf16 {
                    tensor.cast(FloatDType::BF16)
                } else {
                    tensor
                }
            };
            let delta = cast(Tensor::<3>::from_data(
                TensorData::new(data.0.clone(), shapes.0),
                $device,
            ))
            .require_grad();
            let xs = cast(Tensor::<3>::from_data(
                TensorData::new(data.1.clone(), shapes.0),
                $device,
            ))
            .require_grad();
            let b = cast(Tensor::<3>::from_data(
                TensorData::new(data.2.clone(), shapes.1),
                $device,
            ))
            .require_grad();
            let c = cast(Tensor::<3>::from_data(
                TensorData::new(data.3.clone(), shapes.1),
                $device,
            ))
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
            let weights = cast(Tensor::<3>::from_data(
                TensorData::new(weight_data.clone(), shapes.0),
                $device,
            ));
            let mut grads = (y.clone() * weights).sum().backward();
            (
                y.into_data(),
                [
                    delta.grad_remove(&mut grads).unwrap().into_data(),
                    xs.grad_remove(&mut grads).unwrap().into_data(),
                    b.grad_remove(&mut grads).unwrap().into_data(),
                    c.grad_remove(&mut grads).unwrap().into_data(),
                    a.grad_remove(&mut grads).unwrap().into_data(),
                    d.grad_remove(&mut grads).unwrap().into_data(),
                    h.grad_remove(&mut grads).unwrap().into_data(),
                ],
            )
        }};
    }

    let (cpu_y, cpu_grads) = run!(&cpu, false);
    let (gpu_y, gpu_grads) = run!(&gpu, true);
    let y_difference = max_diff(cpu_y, gpu_y);
    assert!(y_difference < 2e-2, "y max diff: {y_difference}");
    for ((name, cpu), gpu) in ["delta", "xs", "b", "c", "a", "d", "h"]
        .into_iter()
        .zip(cpu_grads)
        .zip(gpu_grads)
    {
        let difference = max_diff(cpu, gpu);
        assert!(difference < 2e-2, "{name} gradient max diff: {difference}");
    }
}

#[test]
fn test_cubecl_selective_scan_checkpointed_backward_matches_ndarray() {
    assert_cubecl_selective_scan_backward(3);
}

/// Pins the sweep kernels' geometry guards off the production shape:
/// a channel count spanning two tiles with a partial second tile and a
/// sequence with a partial tail segment.
#[test]
fn test_cubecl_selective_scan_channel_overhang_backward_matches_ndarray() {
    assert_cubecl_selective_scan_backward_geometry(3, 37, 19, 4);
}

/// The remaining supported state width (4 is covered above, 16 by the
/// bf16 and production tests).
#[test]
fn test_cubecl_selective_scan_state_dim_8_backward_matches_ndarray() {
    assert_cubecl_selective_scan_backward_geometry(3, 37, 16, 8);
}

/// Unsupported state widths must refuse loudly instead of silently
/// corrupting: the CUDA runtime displaces writes into state tensors
/// with non-power-of-two minor strides (kernel logic itself is
/// shape-generic — Metal passes any width).
#[test]
#[should_panic(expected = "power-of-two state_dim")]
fn test_gpu_selective_scan_rejects_unsupported_state_dim() {
    assert_cubecl_selective_scan_backward_geometry(3, 37, 16, 5);
}
