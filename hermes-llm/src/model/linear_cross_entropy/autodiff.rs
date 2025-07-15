//! First-order autodiff node for the fused head + loss.

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

use super::LinearCrossEntropyBackend;

#[derive(Clone, Debug)]
struct LinearCrossEntropyState<B: LinearCrossEntropyBackend> {
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    targets: IntTensor<B>,
    logical_vocab_size: usize,
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
            state.logical_vocab_size,
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
        logical_vocab_size: usize,
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
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                );
                let state = LinearCrossEntropyState {
                    hidden: hidden.primitive,
                    weight: weight.primitive,
                    bias: bias.primitive,
                    targets,
                    logical_vocab_size,
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
                    logical_vocab_size,
                    chunk_size,
                    use_bias,
                );
                prep.finish(output)
            }
        }
    }
}
