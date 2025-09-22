//! First-order autodiff node for the selective scan.
//!
//! The tracked forward saves the inputs plus the per-segment checkpoints
//! the backend's fused backward re-derives states from; the backward
//! consumes the output gradient and registers gradients for every parent
//! in one call. Second-order autodiff is intentionally unsupported (the
//! trait's default backward panics).

use burn::backend::tensor::FloatTensor;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};

use super::MambaBackend;

#[derive(Clone, Debug)]
struct SelectiveScanState<B: MambaBackend> {
    delta: FloatTensor<B>,
    xs: FloatTensor<B>,
    b_mat: FloatTensor<B>,
    c_mat: FloatTensor<B>,
    a: FloatTensor<B>,
    d: FloatTensor<B>,
    h: FloatTensor<B>,
    states: FloatTensor<B>,
    state_dim: usize,
}

#[derive(Debug)]
struct SelectiveScanBackward;

impl<B: MambaBackend> Backward<B, 7> for SelectiveScanBackward {
    type State = SelectiveScanState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 7>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let [node_delta, node_xs, node_b, node_c, node_a, node_d, node_h] = ops.parents;
        let grad_y = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let output = B::selective_scan_backward(
            state.delta,
            state.xs,
            state.b_mat,
            state.c_mat,
            state.a,
            state.d,
            state.h,
            state.states,
            grad_y,
            state.state_dim,
        );

        for (node, grad) in [
            (node_delta, output.0),
            (node_xs, output.1),
            (node_b, output.2),
            (node_c, output.3),
            (node_a, output.4),
            (node_d, output.5),
            (node_h, output.6),
        ] {
            if let Some(node) = node {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

impl<B: MambaBackend, C: CheckpointStrategy> MambaBackend for Autodiff<B, C> {
    fn selective_scan_inner(
        delta: FloatTensor<Self>,
        xs: FloatTensor<Self>,
        b_mat: FloatTensor<Self>,
        c_mat: FloatTensor<Self>,
        a: FloatTensor<Self>,
        d: FloatTensor<Self>,
        h: FloatTensor<Self>,
        state_dim: usize,
        _save_states: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        match SelectiveScanBackward
            .prepare::<C>([
                delta.node.clone(),
                xs.node.clone(),
                b_mat.node.clone(),
                c_mat.node.clone(),
                a.node.clone(),
                d.node.clone(),
                h.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (y, h_out, states) = B::selective_scan_inner(
                    delta.primitive.clone(),
                    xs.primitive.clone(),
                    b_mat.primitive.clone(),
                    c_mat.primitive.clone(),
                    a.primitive.clone(),
                    d.primitive.clone(),
                    h.primitive.clone(),
                    state_dim,
                    true,
                );
                let state = SelectiveScanState {
                    delta: delta.primitive,
                    xs: xs.primitive,
                    b_mat: b_mat.primitive,
                    c_mat: c_mat.primitive,
                    a: a.primitive,
                    d: d.primitive,
                    h: h.primitive,
                    states,
                    state_dim,
                };
                (
                    prep.finish(state, y),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out.clone()),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out),
                )
            }
            OpsKind::UnTracked(prep) => {
                let (y, h_out, _states) = B::selective_scan_inner(
                    delta.primitive,
                    xs.primitive,
                    b_mat.primitive,
                    c_mat.primitive,
                    a.primitive,
                    d.primitive,
                    h.primitive,
                    state_dim,
                    false,
                );
                (
                    prep.finish(y),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out.clone()),
                    <Self as burn::backend::AutodiffBackend>::from_inner(h_out),
                )
            }
        }
    }
}
