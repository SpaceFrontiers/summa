//! Memory-linear scaled dot-product attention with an explicit backward.
//!
//! CUDA uses CubeCL kernels. Other backends keep a tensor-op reference so the
//! model has one portable attention API.

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
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

#[cfg(feature = "cuda")]
use super::matmul::{matmul_4, matmul_input};

/// Backend capability used by full-sequence Transformer attention.
#[cfg_attr(feature = "cuda", backend_extension(Cuda, Autodiff))]
#[cfg_attr(
    all(not(feature = "cuda"), feature = "metal"),
    backend_extension(Metal, Autodiff)
)]
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal")),
    backend_extension(NdArray, Autodiff)
)]
pub trait AttentionBackend: Backend {
    #[allow(clippy::type_complexity)]
    fn attention_inner(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        causal: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );

    #[allow(clippy::too_many_arguments, unused_variables)]
    fn attention_backward(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        output: FloatTensor<Self>,
        _stats: FloatTensor<Self>,
        grad_output: FloatTensor<Self>,
        causal: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        panic!("custom attention only supports first-order autodiff")
    }
}

pub(super) fn fused_attention(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    causal: bool,
) -> Tensor<4> {
    let query_heads = query.dims()[1];
    if query.device() == Device::ndarray() {
        return attention_probabilities(query, key, causal).matmul(repeat_kv(value, query_heads));
    }
    let key = repeat_kv(key, query_heads);
    let value = repeat_kv(value, query_heads);
    let output = Dispatch::attention_inner(
        query.into_dispatch(),
        key.into_dispatch(),
        value.into_dispatch(),
        causal,
    );
    Tensor::from_dispatch(output.0)
}

pub(super) fn repeat_kv(tensor: Tensor<4>, query_heads: usize) -> Tensor<4> {
    let [batch, kv_heads, sequence, head_dim] = tensor.dims();
    if kv_heads == query_heads {
        return tensor;
    }
    assert_eq!(query_heads % kv_heads, 0);
    let repeats = query_heads / kv_heads;
    tensor
        .unsqueeze_dim::<5>(2)
        .repeat_dim(2, repeats)
        .reshape([batch, query_heads, sequence, head_dim])
}

fn reduce_kv_grad(tensor: Tensor<4>, kv_heads: usize) -> Tensor<4> {
    let [batch, query_heads, sequence, head_dim] = tensor.dims();
    if kv_heads == query_heads {
        return tensor;
    }
    let repeats = query_heads / kv_heads;
    tensor
        .reshape([batch, kv_heads, repeats, sequence, head_dim])
        .sum_dim(2)
        .reshape([batch, kv_heads, sequence, head_dim])
}

fn causal_mask(sequence: usize, device: &Device) -> Tensor<4, Bool> {
    let mut values = vec![false; sequence * sequence];
    for row in 0..sequence {
        for column in row + 1..sequence {
            values[row * sequence + column] = true;
        }
    }
    Tensor::<2, Bool>::from_data(TensorData::new(values, [sequence, sequence]), device)
        .reshape([1, 1, sequence, sequence])
}

#[cfg(feature = "cuda")]
fn causal_chunk_mask(
    sequence: usize,
    row_start: usize,
    rows: usize,
    device: &Device,
) -> Tensor<4, Bool> {
    let mut values = vec![false; rows * sequence];
    for row in 0..rows {
        let global_row = row_start + row;
        for column in global_row + 1..sequence {
            values[row * sequence + column] = true;
        }
    }
    Tensor::<2, Bool>::from_data(TensorData::new(values, [rows, sequence]), device)
        .reshape([1, 1, rows, sequence])
}

fn attention_probabilities(query: Tensor<4>, key: Tensor<4>, causal: bool) -> Tensor<4> {
    let [_, query_heads, sequence, head_dim] = query.dims();
    let key = repeat_kv(key, query_heads);
    let mut scores = query
        .matmul(key.transpose())
        .div_scalar((head_dim as f32).sqrt());
    if causal {
        let device = scores.device();
        scores = scores.mask_fill(causal_mask(sequence, &device), f32::NEG_INFINITY);
    }
    softmax(scores, 3)
}

#[allow(clippy::type_complexity)]
fn reference_attention<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    causal: bool,
) -> (
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
    let saved_query = query.clone();
    let saved_key = key.clone();
    let saved_value = value.clone();
    let query = Tensor::<4>::from_primitive::<B>(query);
    let key = Tensor::<4>::from_primitive::<B>(key);
    let value = Tensor::<4>::from_primitive::<B>(value);
    let [batch, query_heads, sequence, _] = query.dims();
    let probabilities = attention_probabilities(query, key, causal);
    let output = probabilities.matmul(repeat_kv(value, query_heads));
    let stats = Tensor::<4>::zeros([batch, query_heads, sequence, 1], &output.device());
    let output = output
        .try_into_primitive::<B>()
        .expect("attention output stayed on its input backend");
    (
        output.clone(),
        // The reference backward recomputes softmax. CUDA replaces this with
        // per-row log-sum-exp statistics.
        stats
            .try_into_primitive::<B>()
            .expect("attention stats stayed on its input backend"),
        saved_query,
        saved_key,
        saved_value,
        output,
    )
}

fn reference_attention_backward<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    output: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    causal: bool,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let query = Tensor::<4>::from_primitive::<B>(query);
    let key = Tensor::<4>::from_primitive::<B>(key);
    let value = Tensor::<4>::from_primitive::<B>(value);
    let output = Tensor::<4>::from_primitive::<B>(output);
    let grad_output = Tensor::<4>::from_primitive::<B>(grad_output);
    let [_, query_heads, _, head_dim] = query.dims();
    let kv_heads = key.dims()[1];
    let key_repeated = repeat_kv(key, query_heads);
    let value_repeated = repeat_kv(value, query_heads);
    let probabilities = attention_probabilities(query.clone(), key_repeated.clone(), causal);
    let correction = (grad_output.clone() * output).sum_dim(3);
    let grad_probabilities = grad_output.clone().matmul(value_repeated.transpose());
    let grad_scores = probabilities.clone() * (grad_probabilities - correction);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let grad_query = grad_scores.clone().matmul(key_repeated).mul_scalar(scale);
    let grad_key = reduce_kv_grad(
        grad_scores.transpose().matmul(query).mul_scalar(scale),
        kv_heads,
    );
    let grad_value = reduce_kv_grad(probabilities.transpose().matmul(grad_output), kv_heads);
    (
        grad_query
            .try_into_primitive::<B>()
            .expect("query gradient stayed on its input backend"),
        grad_key
            .try_into_primitive::<B>()
            .expect("key gradient stayed on its input backend"),
        grad_value
            .try_into_primitive::<B>()
            .expect("value gradient stayed on its input backend"),
    )
}

/// Recompute attention a block of query rows at a time. Each block uses the
/// backend's accelerated matmuls while memory remains linear in sequence
/// length for a fixed block size.
#[cfg(feature = "cuda")]
pub(super) fn chunked_attention_backward<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    output: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    causal: bool,
    chunk_size: usize,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let query = Tensor::<4>::from_primitive::<B>(query);
    let key = Tensor::<4>::from_primitive::<B>(key);
    let value = Tensor::<4>::from_primitive::<B>(value);
    let output = Tensor::<4>::from_primitive::<B>(output);
    let grad_output = Tensor::<4>::from_primitive::<B>(grad_output);
    let [batch, heads, sequence, head_dim] = query.dims();
    assert_eq!(key.dims(), [batch, heads, sequence, head_dim]);
    assert_eq!(value.dims(), [batch, heads, sequence, head_dim]);
    assert!(chunk_size > 0);

    let device = query.device();
    let key_transposed = matmul_input(key.clone().transpose());
    let value_transposed = matmul_input(value.clone().transpose());
    let grad_output_compute = matmul_input(grad_output.clone());
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut query_gradients = Vec::with_capacity(sequence.div_ceil(chunk_size));
    let mut key_gradient = Tensor::zeros([batch, heads, sequence, head_dim], &device);
    let mut value_gradient = Tensor::zeros([batch, heads, sequence, head_dim], &device);

    for start in (0..sequence).step_by(chunk_size) {
        let end = (start + chunk_size).min(sequence);
        let rows = end - start;
        let query_chunk = query
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        let output_chunk = output
            .clone()
            .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        let grad_output_chunk =
            grad_output
                .clone()
                .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        let grad_output_compute_chunk =
            grad_output_compute
                .clone()
                .slice([0..batch, 0..heads, start..end, 0..head_dim]);
        let mut scores = matmul_4(query_chunk.clone(), key_transposed.clone()) * scale;
        if causal {
            scores = scores.mask_fill(
                causal_chunk_mask(sequence, start, rows, &device),
                f32::NEG_INFINITY,
            );
        }
        let probabilities = softmax(scores, 3);
        let correction = (grad_output_chunk.clone() * output_chunk).sum_dim(3);
        let probability_gradient =
            matmul_4(grad_output_compute_chunk.clone(), value_transposed.clone());
        let score_gradient = probabilities.clone() * (probability_gradient - correction) * scale;
        let score_gradient_compute = matmul_input(score_gradient);

        query_gradients.push(matmul_4(score_gradient_compute.clone(), key.clone()));
        key_gradient = key_gradient + matmul_4(score_gradient_compute.transpose(), query_chunk);
        value_gradient =
            value_gradient + matmul_4(probabilities.transpose(), grad_output_compute_chunk);
    }

    (
        Tensor::cat(query_gradients, 2)
            .try_into_primitive::<B>()
            .expect("query gradient stayed on its input backend"),
        key_gradient
            .try_into_primitive::<B>()
            .expect("key gradient stayed on its input backend"),
        value_gradient
            .try_into_primitive::<B>()
            .expect("value gradient stayed on its input backend"),
    )
}

macro_rules! impl_reference_attention {
    ($backend:ty) => {
        impl AttentionBackend for $backend {
            fn attention_inner(
                query: FloatTensor<Self>,
                key: FloatTensor<Self>,
                value: FloatTensor<Self>,
                causal: bool,
            ) -> (
                FloatTensor<Self>,
                FloatTensor<Self>,
                FloatTensor<Self>,
                FloatTensor<Self>,
                FloatTensor<Self>,
                FloatTensor<Self>,
            ) {
                reference_attention::<Self>(query, key, value, causal)
            }

            fn attention_backward(
                query: FloatTensor<Self>,
                key: FloatTensor<Self>,
                value: FloatTensor<Self>,
                output: FloatTensor<Self>,
                _stats: FloatTensor<Self>,
                grad_output: FloatTensor<Self>,
                causal: bool,
            ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
                reference_attention_backward::<Self>(query, key, value, output, grad_output, causal)
            }
        }
    };
}

impl_reference_attention!(burn_ndarray::NdArray);

#[cfg(feature = "metal")]
impl_reference_attention!(burn_wgpu::Metal);

#[derive(Clone, Debug)]
struct AttentionState<B: AttentionBackend> {
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    output: FloatTensor<B>,
    stats: FloatTensor<B>,
    causal: bool,
}

#[derive(Debug)]
struct AttentionBackward;

impl<B: AttentionBackend> Backward<B, 3> for AttentionBackward {
    type State = AttentionState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_query, node_key, node_value] = ops.parents;
        let grad_output = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let output = B::attention_backward(
            state.query,
            state.key,
            state.value,
            state.output,
            state.stats,
            grad_output,
            state.causal,
        );
        for (node, grad) in [
            (node_query, output.0),
            (node_key, output.1),
            (node_value, output.2),
        ] {
            if let Some(node) = node {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

impl<B: AttentionBackend, C: CheckpointStrategy> AttentionBackend for Autodiff<B, C> {
    fn attention_inner(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        causal: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        match AttentionBackward
            .prepare::<C>([query.node.clone(), key.node.clone(), value.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (output, stats, saved_query, saved_key, saved_value, saved_output) =
                    B::attention_inner(
                        query.primitive.clone(),
                        key.primitive.clone(),
                        value.primitive.clone(),
                        causal,
                    );
                let state = AttentionState {
                    query: saved_query.clone(),
                    key: saved_key.clone(),
                    value: saved_value.clone(),
                    output: saved_output.clone(),
                    stats: stats.clone(),
                    causal,
                };
                (
                    prep.finish(state, output),
                    <Self as burn::backend::AutodiffBackend>::from_inner(stats),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_query),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_key),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_value),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_output),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (output, stats, saved_query, saved_key, saved_value, saved_output) =
                    B::attention_inner(query.primitive, key.primitive, value.primitive, causal);
                (
                    prep.finish(output),
                    <Self as burn::backend::AutodiffBackend>::from_inner(stats),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_query),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_key),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_value),
                    <Self as burn::backend::AutodiffBackend>::from_inner(saved_output),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::{attention_probabilities, fused_attention, repeat_kv};
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    use crate::model::test_support::snapshot;

    fn values(length: usize, scale: f32) -> Vec<f32> {
        (0..length)
            .map(|index| (index as f32 * scale).sin() * 0.25)
            .collect()
    }

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
    fn custom_attention_backward_matches_burn_autodiff_for_causal_gqa() {
        let device = Device::ndarray().autodiff();
        let (batch, query_heads, kv_heads, sequence, head_dim) = (2, 4, 2, 5, 4);
        let query_data = values(batch * query_heads * sequence * head_dim, 0.071);
        let key_data = values(batch * kv_heads * sequence * head_dim, 0.097);
        let value_data = values(batch * kv_heads * sequence * head_dim, 0.113);
        let factor_data = values(batch * query_heads * sequence * head_dim, 0.137);

        let run = |custom: bool| {
            let query = Tensor::<4>::from_data(
                TensorData::new(query_data.clone(), [batch, query_heads, sequence, head_dim]),
                &device,
            )
            .require_grad();
            let key = Tensor::<4>::from_data(
                TensorData::new(key_data.clone(), [batch, kv_heads, sequence, head_dim]),
                &device,
            )
            .require_grad();
            let value = Tensor::<4>::from_data(
                TensorData::new(value_data.clone(), [batch, kv_heads, sequence, head_dim]),
                &device,
            )
            .require_grad();
            let output = if custom {
                fused_attention(query.clone(), key.clone(), value.clone(), true)
            } else {
                attention_probabilities(query.clone(), key.clone(), true)
                    .matmul(repeat_kv(value.clone(), query_heads))
            };
            let output_data = output.clone().into_data();
            let factors = Tensor::<4>::from_data(
                TensorData::new(
                    factor_data.clone(),
                    [batch, query_heads, sequence, head_dim],
                ),
                &device,
            );
            let mut gradients = (output * factors).sum().backward();
            (
                output_data,
                query.grad_remove(&mut gradients).unwrap().into_data(),
                key.grad_remove(&mut gradients).unwrap().into_data(),
                value.grad_remove(&mut gradients).unwrap().into_data(),
            )
        };

        let expected = run(false);
        let actual = run(true);
        assert!(max_diff(expected.0, actual.0) < 1e-6);
        assert!(max_diff(expected.1, actual.1) < 1e-5);
        assert!(max_diff(expected.2, actual.2) < 1e-5);
        assert!(max_diff(expected.3, actual.3) < 1e-5);
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_attention_matches_cpu_reference_for_causal_gqa() {
        let cpu_device = Device::ndarray().autodiff();
        let cuda_device = Device::cuda(0).autodiff();
        let (batch, query_heads, kv_heads, sequence, head_dim) = (1, 4, 2, 64, 64);
        let query_data = values(batch * query_heads * sequence * head_dim, 0.071);
        let key_data = values(batch * kv_heads * sequence * head_dim, 0.097);
        let value_data = values(batch * kv_heads * sequence * head_dim, 0.113);
        let factor_data = values(batch * query_heads * sequence * head_dim, 0.137);

        let cpu_query = Tensor::<4>::from_data(
            TensorData::new(query_data.clone(), [batch, query_heads, sequence, head_dim]),
            &cpu_device,
        )
        .require_grad();
        let cpu_key = Tensor::<4>::from_data(
            TensorData::new(key_data.clone(), [batch, kv_heads, sequence, head_dim]),
            &cpu_device,
        )
        .require_grad();
        let cpu_value = Tensor::<4>::from_data(
            TensorData::new(value_data.clone(), [batch, kv_heads, sequence, head_dim]),
            &cpu_device,
        )
        .require_grad();
        let cpu_output = attention_probabilities(cpu_query.clone(), cpu_key.clone(), true)
            .matmul(repeat_kv(cpu_value.clone(), query_heads));
        let cpu_output_data = snapshot(cpu_output.clone());
        let cpu_factors = Tensor::<4>::from_data(
            TensorData::new(
                factor_data.clone(),
                [batch, query_heads, sequence, head_dim],
            ),
            &cpu_device,
        );
        let mut cpu_gradients = (cpu_output * cpu_factors).sum().backward();
        let expected = (
            cpu_output_data,
            cpu_query
                .grad_remove(&mut cpu_gradients)
                .unwrap()
                .into_data(),
            cpu_key.grad_remove(&mut cpu_gradients).unwrap().into_data(),
            cpu_value
                .grad_remove(&mut cpu_gradients)
                .unwrap()
                .into_data(),
        );

        let cuda_query = Tensor::<4>::from_data(
            TensorData::new(query_data, [batch, query_heads, sequence, head_dim]),
            &cuda_device,
        )
        .require_grad();
        let cuda_key = Tensor::<4>::from_data(
            TensorData::new(key_data, [batch, kv_heads, sequence, head_dim]),
            &cuda_device,
        )
        .require_grad();
        let cuda_value = Tensor::<4>::from_data(
            TensorData::new(value_data, [batch, kv_heads, sequence, head_dim]),
            &cuda_device,
        )
        .require_grad();
        let cuda_output = fused_attention(
            cuda_query.clone(),
            cuda_key.clone(),
            cuda_value.clone(),
            true,
        );
        let cuda_output_data = snapshot(cuda_output.clone());
        let cuda_factors = Tensor::<4>::from_data(
            TensorData::new(factor_data, [batch, query_heads, sequence, head_dim]),
            &cuda_device,
        );
        let mut cuda_gradients = (cuda_output * cuda_factors).sum().backward();
        let actual = (
            cuda_output_data,
            cuda_query
                .grad_remove(&mut cuda_gradients)
                .unwrap()
                .into_data(),
            cuda_key
                .grad_remove(&mut cuda_gradients)
                .unwrap()
                .into_data(),
            cuda_value
                .grad_remove(&mut cuda_gradients)
                .unwrap()
                .into_data(),
        );

        assert!(max_diff(expected.0, actual.0) < 0.01);
        assert!(max_diff(expected.1, actual.1) < 0.02);
        assert!(max_diff(expected.2, actual.2) < 0.02);
        assert!(max_diff(expected.3, actual.3) < 0.02);
    }
}
