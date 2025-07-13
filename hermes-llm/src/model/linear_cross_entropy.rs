//! Memory-bounded output projection and cross-entropy for training.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
use burn::backend::{
    Backend, Dispatch, DispatchKindConversion, DispatchTensor, NdArray, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{IndexingUpdateOp, Int};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_nn::loss::CrossEntropyLossConfig;

use super::matmul::{matmul_2, matmul_input};

/// Backend capability for training without retaining full-vocabulary logits.
#[backend_extension(
    Cuda: cfg(feature = "cuda"),
    Metal: cfg(feature = "metal"),
    NdArray,
    Autodiff,
)]
pub trait LinearCrossEntropyBackend: Backend {
    fn linear_cross_entropy_inner(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: IntTensor<Self>,
        chunk_size: usize,
        use_bias: bool,
    ) -> FloatTensor<Self>;

    #[allow(unused_variables)]
    fn linear_cross_entropy_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: IntTensor<Self>,
        grad_output: FloatTensor<Self>,
        chunk_size: usize,
        use_bias: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        panic!("linear cross-entropy only supports first-order autodiff")
    }
}

pub(super) fn linear_cross_entropy(
    hidden: Tensor<2>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
    targets: Tensor<1, Int>,
    chunk_size: usize,
) -> Tensor<1> {
    assert!(
        chunk_size > 0,
        "linear cross-entropy chunk size must be positive"
    );
    let use_bias = bias.is_some();
    let bias = bias.unwrap_or_else(|| Tensor::zeros([weight.dims()[0]], &hidden.device()));
    let output = Dispatch::linear_cross_entropy_inner(
        hidden.into_dispatch(),
        weight.into_dispatch(),
        bias.into_dispatch(),
        targets.into_dispatch(),
        chunk_size,
        use_bias,
    );
    Tensor::from_dispatch(output)
}

fn chunked_loss<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    chunk_size: usize,
    use_bias: bool,
) -> FloatTensor<B>
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let hidden = Tensor::<2>::from_primitive::<B>(hidden);
    let weight = Tensor::<2>::from_primitive::<B>(weight);
    let bias = Tensor::<1>::from_primitive::<B>(bias);
    let targets = Tensor::<1, Int>::from_primitive::<B>(targets);
    let [tokens, hidden_size] = hidden.dims();
    let [vocab_size, weight_hidden] = weight.dims();
    assert_eq!(hidden_size, weight_hidden);
    assert_eq!(targets.dims(), [tokens]);
    assert_eq!(bias.dims(), [vocab_size]);

    let criterion = CrossEntropyLossConfig::new().init(&hidden.device());
    let weight_transposed = matmul_input(weight.transpose());
    let mut total = None;
    for start in (0..tokens).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens);
        let logits = matmul_2(
            hidden.clone().slice([start..end, 0..hidden_size]),
            weight_transposed.clone(),
        );
        let logits = if use_bias {
            logits + bias.clone().reshape([1, vocab_size])
        } else {
            logits
        };
        let loss = criterion.forward(logits, targets.clone().slice(start..end));
        let loss = loss.mul_scalar((end - start) as f32);
        total = Some(match total {
            Some(total) => total + loss,
            None => loss,
        });
    }
    total
        .expect("linear cross-entropy requires at least one token")
        .div_scalar(tokens as f32)
        .try_into_primitive::<B>()
        .expect("linear cross-entropy output stayed on its input backend")
}

fn chunked_backward<B: Backend>(
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    grad_output: FloatTensor<B>,
    chunk_size: usize,
    use_bias: bool,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let hidden = Tensor::<2>::from_primitive::<B>(hidden);
    let weight = Tensor::<2>::from_primitive::<B>(weight);
    let bias = Tensor::<1>::from_primitive::<B>(bias);
    let targets = Tensor::<1, Int>::from_primitive::<B>(targets);
    let grad_output = Tensor::<1>::from_primitive::<B>(grad_output).reshape([1, 1]);
    let [tokens, hidden_size] = hidden.dims();
    let [vocab_size, weight_hidden] = weight.dims();
    assert_eq!(hidden_size, weight_hidden);
    let device = hidden.device();
    let scale = grad_output.div_scalar(tokens as f32);
    let weight = matmul_input(weight);
    let weight_transposed = weight.clone().transpose();
    let mut hidden_gradients = Vec::with_capacity(tokens.div_ceil(chunk_size));
    let mut weight_gradient = Tensor::<2>::zeros([vocab_size, hidden_size], &device);
    let mut bias_gradient = Tensor::<1>::zeros([vocab_size], &device);

    for start in (0..tokens).step_by(chunk_size) {
        let end = (start + chunk_size).min(tokens);
        let chunk_tokens = end - start;
        let hidden_chunk = hidden.clone().slice([start..end, 0..hidden_size]);
        let logits = matmul_2(hidden_chunk.clone(), weight_transposed.clone());
        let logits = if use_bias {
            logits + bias.clone().reshape([1, vocab_size])
        } else {
            logits
        };
        let target_indices = targets.clone().slice(start..end).reshape([chunk_tokens, 1]);
        let corrections = Tensor::<2>::ones([chunk_tokens, 1], &device).neg();
        let logits_gradient =
            softmax(logits, 1).scatter(1, target_indices, corrections, IndexingUpdateOp::Add)
                * scale.clone();
        let logits_gradient_compute = matmul_input(logits_gradient.clone());
        hidden_gradients.push(matmul_2(logits_gradient_compute.clone(), weight.clone()));
        weight_gradient =
            weight_gradient + matmul_2(logits_gradient_compute.transpose(), hidden_chunk);
        if use_bias {
            bias_gradient = bias_gradient + logits_gradient.sum_dim(0).reshape([vocab_size]);
        }
    }

    (
        Tensor::cat(hidden_gradients, 0)
            .try_into_primitive::<B>()
            .expect("hidden gradient stayed on its input backend"),
        weight_gradient
            .try_into_primitive::<B>()
            .expect("weight gradient stayed on its input backend"),
        bias_gradient
            .try_into_primitive::<B>()
            .expect("bias gradient stayed on its input backend"),
    )
}

macro_rules! impl_reference_linear_cross_entropy {
    ($backend:ty) => {
        impl LinearCrossEntropyBackend for $backend {
            fn linear_cross_entropy_inner(
                hidden: FloatTensor<Self>,
                weight: FloatTensor<Self>,
                bias: FloatTensor<Self>,
                targets: IntTensor<Self>,
                chunk_size: usize,
                use_bias: bool,
            ) -> FloatTensor<Self> {
                chunked_loss::<Self>(hidden, weight, bias, targets, chunk_size, use_bias)
            }

            fn linear_cross_entropy_backward(
                hidden: FloatTensor<Self>,
                weight: FloatTensor<Self>,
                bias: FloatTensor<Self>,
                targets: IntTensor<Self>,
                grad_output: FloatTensor<Self>,
                chunk_size: usize,
                use_bias: bool,
            ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
                chunked_backward::<Self>(
                    hidden,
                    weight,
                    bias,
                    targets,
                    grad_output,
                    chunk_size,
                    use_bias,
                )
            }
        }
    };
}

impl_reference_linear_cross_entropy!(burn_ndarray::NdArray);

#[cfg(feature = "metal")]
impl_reference_linear_cross_entropy!(burn_wgpu::Metal);

#[cfg(feature = "cuda")]
impl_reference_linear_cross_entropy!(burn_cuda::Cuda);

#[derive(Clone, Debug)]
struct LinearCrossEntropyState<B: LinearCrossEntropyBackend> {
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    chunk_size: usize,
    use_bias: bool,
}

#[derive(Debug)]
struct LinearCrossEntropyBackward;

impl<B: LinearCrossEntropyBackend> Backward<B, 3> for LinearCrossEntropyBackward {
    type State = LinearCrossEntropyState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_hidden, node_weight, node_bias] = ops.parents;
        let grad_output = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let gradients = B::linear_cross_entropy_backward(
            state.hidden,
            state.weight,
            state.bias,
            state.targets,
            grad_output,
            state.chunk_size,
            state.use_bias,
        );
        for (node, gradient) in [
            (node_hidden, gradients.0),
            (node_weight, gradients.1),
            (node_bias, gradients.2),
        ] {
            if let Some(node) = node {
                grads.register::<B>(node.id, gradient);
            }
        }
    }
}

impl<B: LinearCrossEntropyBackend, C: CheckpointStrategy> LinearCrossEntropyBackend
    for Autodiff<B, C>
{
    fn linear_cross_entropy_inner(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        targets: IntTensor<Self>,
        chunk_size: usize,
        use_bias: bool,
    ) -> FloatTensor<Self> {
        match LinearCrossEntropyBackward
            .prepare::<C>([hidden.node.clone(), weight.node.clone(), bias.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let targets = <Self as burn::backend::AutodiffBackend>::int_inner(targets);
                let output = B::linear_cross_entropy_inner(
                    hidden.primitive.clone(),
                    weight.primitive.clone(),
                    bias.primitive.clone(),
                    targets.clone(),
                    chunk_size,
                    use_bias,
                );
                let state = LinearCrossEntropyState {
                    hidden: hidden.primitive,
                    weight: weight.primitive,
                    bias: bias.primitive,
                    targets,
                    chunk_size,
                    use_bias,
                };
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                let targets = <Self as burn::backend::AutodiffBackend>::int_inner(targets);
                let output = B::linear_cross_entropy_inner(
                    hidden.primitive,
                    weight.primitive,
                    bias.primitive,
                    targets,
                    chunk_size,
                    use_bias,
                );
                prep.finish(output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

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
        let (tokens, hidden_size, vocab_size) = (7, 5, 11);
        let hidden_data = (0..tokens * hidden_size)
            .map(|index| (index as f32 * 0.071).sin() * 0.2)
            .collect::<Vec<_>>();
        let weight_data = (0..vocab_size * hidden_size)
            .map(|index| (index as f32 * 0.113).cos() * 0.3)
            .collect::<Vec<_>>();
        let bias_data = (0..vocab_size)
            .map(|index| index as f32 * 0.01)
            .collect::<Vec<_>>();
        let target_data = vec![0_i64, 3, 7, 2, 10, 4, 1];

        let run = |chunked: bool, use_bias: bool| {
            let hidden = Tensor::<2>::from_data(
                TensorData::new(hidden_data.clone(), [tokens, hidden_size]),
                &device,
            )
            .require_grad();
            let weight = Tensor::<2>::from_data(
                TensorData::new(weight_data.clone(), [vocab_size, hidden_size]),
                &device,
            )
            .require_grad();
            let bias =
                Tensor::<1>::from_data(TensorData::new(bias_data.clone(), [vocab_size]), &device)
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
                    3,
                )
            } else {
                let logits = hidden.clone().matmul(weight.clone().transpose());
                let logits = if use_bias {
                    logits + bias.clone().reshape([1, vocab_size])
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
