use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use burn::module::{Module, ModuleMapper, Param, ParamId};
#[cfg(feature = "cuda")]
use burn::tensor::FloatDType;
use burn::tensor::Tensor;
use burn_optim::GradientsParams;
use hermes_llm::Backend;

use crate::TrainBackend;

const MOMENTUM: f64 = 0.95;
const NS_COEFFICIENTS: (f64, f64, f64) = (3.4445, -4.775, 2.0315);
const NS_STEPS: usize = 5;
const EPSILON: f64 = 1e-7;

/// Muon with Burn's update and hyperparameters, batched by matrix shape.
///
/// Burn's generic optimizer visits every parameter separately. Transformer
/// blocks repeat a small set of matrix shapes, so batching those matrices
/// avoids thousands of tiny GPU launches without changing optimizer state or
/// hyperparameters. CUDA runs Newton-Schulz in BF16, its intended stable
/// compute dtype, while parameters and momentum remain FP32.
pub struct BatchedMuon {
    parameter_ids: Vec<ParamId>,
    velocities: BTreeMap<[usize; 2], Tensor<Backend, 3>>,
}

impl BatchedMuon {
    pub fn new(parameter_ids: Vec<ParamId>) -> Self {
        Self {
            parameter_ids,
            velocities: BTreeMap::new(),
        }
    }

    pub fn step<M: Module<TrainBackend>>(
        &mut self,
        lr: f64,
        model: M,
        mut grads: GradientsParams,
    ) -> Result<M> {
        let mut batches = BTreeMap::<[usize; 2], Vec<(ParamId, Tensor<Backend, 2>)>>::new();
        for id in &self.parameter_ids {
            let grad = grads
                .remove::<Backend, 2>(*id)
                .with_context(|| format!("Muon gradient is missing for parameter {id}"))?;
            batches.entry(grad.dims()).or_default().push((*id, grad));
        }
        ensure!(
            grads.is_empty(),
            "Muon received {} unexpected gradients",
            grads.len()
        );

        let mut updates = GradientsParams::new();
        for (shape, batch) in batches {
            let (ids, gradients): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
            let gradients = Tensor::stack::<3>(gradients, 0);

            let velocity = match self.velocities.remove(&shape) {
                Some(velocity) => gradients.clone() + velocity.mul_scalar(MOMENTUM),
                None => gradients.clone(),
            };
            let momentum_update = velocity.clone().mul_scalar(MOMENTUM) + gradients;
            let orthogonal = zeropower_via_newton_schulz(momentum_update);
            let adjusted_lr = lr * ((shape[0] as f64 / shape[1] as f64).max(1.0)).sqrt();
            let deltas = orthogonal.mul_scalar(adjusted_lr);

            for (index, id) in ids.into_iter().enumerate() {
                let delta = deltas
                    .clone()
                    .slice([index..index + 1, 0..shape[0], 0..shape[1]])
                    .reshape(shape);
                updates.register::<Backend, 2>(id, delta);
            }
            self.velocities.insert(shape, velocity);
        }

        ensure!(
            !self.velocities.is_empty(),
            "Muon has no matrix groups to optimize"
        );
        let mut mapper = MuonUpdateMapper {
            updates: &mut updates,
        };
        let model = model.map(&mut mapper);
        ensure!(
            updates.is_empty(),
            "{} Muon updates did not match model parameters",
            updates.len()
        );
        Ok(model)
    }
}

fn zeropower_via_newton_schulz(gradient: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    let [_, rows, columns] = gradient.dims();
    let (mut x, transpose) = if rows > columns {
        (gradient.swap_dims(1, 2), true)
    } else {
        (gradient, false)
    };
    x = to_compute_dtype(x);
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(EPSILON);
    x = x / norm;

    let (a, b, c) = NS_COEFFICIENTS;
    for _ in 0..NS_STEPS {
        let gram = x.clone().matmul(x.clone().swap_dims(1, 2));
        let polynomial = gram.clone().mul_scalar(b) + gram.clone().matmul(gram).mul_scalar(c);
        x = x.clone().mul_scalar(a) + polynomial.matmul(x);
    }

    x = from_compute_dtype(x);
    if transpose { x.swap_dims(1, 2) } else { x }
}

#[cfg(feature = "cuda")]
fn to_compute_dtype(tensor: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    tensor.cast(FloatDType::BF16)
}

#[cfg(not(feature = "cuda"))]
fn to_compute_dtype(tensor: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    tensor
}

#[cfg(feature = "cuda")]
fn from_compute_dtype(tensor: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    tensor.cast(FloatDType::F32)
}

#[cfg(not(feature = "cuda"))]
fn from_compute_dtype(tensor: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    tensor
}

struct MuonUpdateMapper<'a> {
    updates: &'a mut GradientsParams,
}

impl ModuleMapper<TrainBackend> for MuonUpdateMapper<'_> {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<TrainBackend, D>>,
    ) -> Param<Tensor<TrainBackend, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = match self.updates.remove::<Backend, D>(id) {
            Some(delta) => {
                let requires_grad = tensor.is_require_grad();
                let mut updated = Tensor::from_inner(tensor.inner() - delta);
                if requires_grad {
                    updated = updated.require_grad();
                }
                updated
            }
            None => tensor,
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use burn::tensor::{TensorData, backend::Backend as _};
    use burn_optim::{MuonConfig, Optimizer};

    use super::*;

    #[derive(Module, Debug)]
    struct MatrixPair<B: burn::tensor::backend::Backend> {
        first: Param<Tensor<B, 2>>,
        second: Param<Tensor<B, 2>>,
    }

    impl<B: burn::tensor::backend::Backend> MatrixPair<B> {
        fn loss(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            (input.clone().matmul(self.first.val()).square()
                + input.matmul(self.second.val()).square())
            .sum()
        }
    }

    fn values(model: &MatrixPair<TrainBackend>) -> Vec<f32> {
        [model.first.val(), model.second.val()]
            .into_iter()
            .flat_map(|tensor| tensor.inner().into_data().to_vec::<f32>().unwrap())
            .collect()
    }

    #[test]
    fn batched_muon_matches_burn_for_repeated_shapes() {
        let device = hermes_llm::default_device();
        Backend::seed(&device, 17);
        let matrix = |scale: f32| {
            Param::from_tensor(
                Tensor::<TrainBackend, 2>::from_data(
                    TensorData::new(
                        (0..24)
                            .map(|i| (i as f32 * scale).sin())
                            .collect::<Vec<_>>(),
                        [4, 6],
                    ),
                    &device,
                )
                .require_grad(),
            )
        };
        let mut actual = MatrixPair {
            first: matrix(0.17),
            second: matrix(0.23),
        };
        let mut expected = actual.clone();
        let ids = vec![actual.first.id, actual.second.id];
        let input = || {
            Tensor::<TrainBackend, 2>::from_data(
                TensorData::new((0..12).map(|i| i as f32 * 0.03).collect(), [3, 4]),
                &device,
            )
        };

        let mut batched = BatchedMuon::new(ids);
        let mut burn = MuonConfig::new().init();
        for _ in 0..2 {
            let grads = GradientsParams::from_grads(actual.loss(input()).backward(), &actual);
            let reference_grads =
                GradientsParams::from_grads(expected.loss(input()).backward(), &expected);
            actual = batched.step(2e-2, actual, grads).unwrap();
            expected = burn.step(2e-2, expected, reference_grads);
        }

        let max_diff = values(&actual)
            .into_iter()
            .zip(values(&expected))
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 2e-5, "Muon parameter max diff: {max_diff}");
    }
}
