//! Parity and padding-contract tests for the fused head + loss.

use burn::tensor::{Device, TensorData};
use burn_nn::loss::CrossEntropyLossConfig;

use super::*;

fn max_diff(lhs: TensorData, rhs: TensorData) -> f32 {
    lhs.convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(rhs.convert::<f32>().to_vec::<f32>().unwrap())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

#[test]
fn chunked_loss_and_gradients_match_materialized_cross_entropy() {
    let device = Device::ndarray().autodiff();
    let (tokens, hidden_size, logical_vocab_size) = (7, 5, 11);
    let hidden_data = (0..tokens * hidden_size)
        .map(|index| (index as f32 * 0.071).sin() * 0.2)
        .collect::<Vec<_>>();
    let target_data = vec![0_i64, 3, 7, 2, 10, 4, 1];

    for stored_vocab_size in [logical_vocab_size, 16] {
        let weight_data = (0..stored_vocab_size * hidden_size)
            .map(|index| (index as f32 * 0.113).cos() * 0.3)
            .collect::<Vec<_>>();
        let bias_data = (0..stored_vocab_size)
            .map(|index| index as f32 * 0.01)
            .collect::<Vec<_>>();

        let run = |chunked: bool, use_bias: bool| {
            let hidden = Tensor::<2>::from_data(
                TensorData::new(hidden_data.clone(), [tokens, hidden_size]),
                &device,
            )
            .require_grad();
            let weight = Tensor::<2>::from_data(
                TensorData::new(weight_data.clone(), [stored_vocab_size, hidden_size]),
                &device,
            )
            .require_grad();
            let bias = Tensor::<1>::from_data(
                TensorData::new(bias_data.clone(), [stored_vocab_size]),
                &device,
            )
            .require_grad();
            let targets = Tensor::<1, Int>::from_data(
                TensorData::new(target_data.clone(), [tokens]),
                &device,
            );
            let loss = if chunked {
                linear_cross_entropy(
                    hidden.clone(),
                    weight.clone(),
                    use_bias.then(|| bias.clone()),
                    targets,
                    logical_vocab_size,
                    3,
                )
            } else {
                let weight = weight
                    .clone()
                    .slice([0..logical_vocab_size, 0..hidden_size]);
                let logits = hidden.clone().matmul(weight.transpose());
                let logits = if use_bias {
                    logits
                        + bias
                            .clone()
                            .slice(0..logical_vocab_size)
                            .reshape([1, logical_vocab_size])
                } else {
                    logits
                };
                CrossEntropyLossConfig::new()
                    .init(&device)
                    .forward(logits, targets)
            };
            let loss_data = loss.clone().into_data();
            let mut gradients = loss.backward();
            (
                loss_data,
                hidden.grad_remove(&mut gradients).unwrap().into_data(),
                weight.grad_remove(&mut gradients).unwrap().into_data(),
                bias.grad_remove(&mut gradients).map(Tensor::into_data),
            )
        };

        for use_bias in [false, true] {
            let expected = run(false, use_bias);
            let actual = run(true, use_bias);
            if stored_vocab_size > logical_vocab_size {
                let padded_weight_gradient = actual
                    .2
                    .clone()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .skip(logical_vocab_size * hidden_size);
                assert!(padded_weight_gradient.into_iter().all(|value| value == 0.0));
                if let Some(bias_gradient) = &actual.3 {
                    let padded_bias_gradient = bias_gradient
                        .clone()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .skip(logical_vocab_size);
                    assert!(padded_bias_gradient.into_iter().all(|value| value == 0.0));
                }
            }
            assert!(max_diff(expected.0, actual.0) < 1e-6);
            assert!(max_diff(expected.1, actual.1) < 1e-6);
            assert!(max_diff(expected.2, actual.2) < 1e-6);
            match (expected.3, actual.3) {
                (Some(expected), Some(actual)) => assert!(max_diff(expected, actual) < 1e-6),
                (None, None) => {}
                _ => panic!("bias gradient tracking differs"),
            }
        }
    }
}

/// GPU fused kernels vs the NdArray reference: padding below the reduction
/// width (empty lanes), bias on/off, and a chunk boundary offset.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn gpu_fused_loss_and_gradients_match_cpu_reference() {
    fn gpu_device() -> Device {
        #[cfg(feature = "metal")]
        return Device::metal(burn::tensor::DeviceKind::DefaultDevice);

        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        return Device::cuda(0);
    }

    let (tokens, hidden_size, stored_vocab, logical_vocab, chunk) = (7, 8, 64, 50, 3);
    let hidden_data: Vec<f32> = (0..tokens * hidden_size)
        .map(|i| ((i * 37 + 11) % 23) as f32 * 0.11 - 1.2)
        .collect();
    let weight_data: Vec<f32> = (0..stored_vocab * hidden_size)
        .map(|i| ((i * 29 + 5) % 19) as f32 * 0.07 - 0.6)
        .collect();
    let bias_data: Vec<f32> = (0..stored_vocab)
        .map(|i| ((i * 13 + 3) % 17) as f32 * 0.05 - 0.4)
        .collect();
    let target_data: Vec<i64> = (0..tokens)
        .map(|i| ((i * 7 + 2) % logical_vocab) as i64)
        .collect();

    for use_bias in [false, true] {
        let mut outputs = Vec::new();
        for device in [Device::ndarray().autodiff(), gpu_device().autodiff()] {
            let hidden = Tensor::<2>::from_data(
                TensorData::new(hidden_data.clone(), [tokens, hidden_size]),
                &device,
            )
            .require_grad();
            let weight = Tensor::<2>::from_data(
                TensorData::new(weight_data.clone(), [stored_vocab, hidden_size]),
                &device,
            )
            .require_grad();
            let bias =
                Tensor::<1>::from_data(TensorData::new(bias_data.clone(), [stored_vocab]), &device)
                    .require_grad();
            let targets = Tensor::<1, Int>::from_data(
                TensorData::new(target_data.clone(), [tokens]),
                &device,
            );
            let loss = linear_cross_entropy(
                hidden.clone(),
                weight.clone(),
                use_bias.then(|| bias.clone()),
                targets,
                logical_vocab,
                chunk,
            );
            let grads = loss.backward();
            outputs.push((
                loss.into_data(),
                hidden.grad(&grads).unwrap().into_data(),
                weight.grad(&grads).unwrap().into_data(),
                bias.grad(&grads).map(|grad| grad.into_data()),
            ));
        }
        let gpu = outputs.pop().unwrap();
        let cpu = outputs.pop().unwrap();
        let loss_diff = max_diff(cpu.0.clone(), gpu.0.clone());
        let hidden_diff = max_diff(cpu.1.clone(), gpu.1.clone());
        let weight_diff = max_diff(cpu.2.clone(), gpu.2.clone());
        eprintln!(
            "use_bias={use_bias} loss_diff={loss_diff:.3e} hidden_diff={hidden_diff:.3e} weight_diff={weight_diff:.3e}"
        );
        // CUDA runs the chunk matmuls on BF16 tensor cores (like real
        // training), so every comparison against the f32 CPU reference
        // carries BF16 rounding; Metal computes in f32 and lands ~1e-7.
        assert!(loss_diff < 1e-2);
        // Gradients flow through BF16 tensor-core matmuls on CUDA, so the
        // comparison against the f32 CPU reference uses the same 1e-2
        // bound as the wider training parity checks.
        assert!(max_diff(cpu.1, gpu.1) < 1e-2);
        assert!(max_diff(cpu.2.clone(), gpu.2.clone()) < 1e-2);
        match (cpu.3, gpu.3) {
            (Some(cpu_bias), Some(gpu_bias)) => assert!(max_diff(cpu_bias, gpu_bias) < 1e-2),
            (None, None) => {}
            _ => panic!("bias gradient presence diverged between devices"),
        }
        // Padded rows must receive exactly zero gradient on the GPU path.
        let padded =
            gpu.2.convert::<f32>().to_vec::<f32>().unwrap()[logical_vocab * hidden_size..].to_vec();
        assert!(padded.into_iter().all(|value| value == 0.0));
    }
}
