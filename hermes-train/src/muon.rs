use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
#[cfg(feature = "cuda")]
use burn::tensor::FloatDType;
use burn::tensor::{Device, Tensor, TensorData};
use burn_optim::GradientsParams;
use burn_pack::{Bytes, DType, Reader, Tensor as PackedTensor, Writer};

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
#[derive(Clone)]
pub struct BatchedMuon {
    parameter_ids: Vec<ParamId>,
    velocities: BTreeMap<[usize; 2], Tensor<3>>,
}

impl BatchedMuon {
    pub fn new(parameter_ids: Vec<ParamId>) -> Self {
        Self {
            parameter_ids,
            velocities: BTreeMap::new(),
        }
    }

    pub fn set_parameter_ids(&mut self, parameter_ids: Vec<ParamId>) {
        self.parameter_ids = parameter_ids;
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensors = self
            .velocities
            .iter()
            .map(|([rows, columns], velocity)| {
                let data = velocity.clone().into_data();
                PackedTensor::new(
                    format!("{rows}x{columns}"),
                    data.dtype,
                    data.shape,
                    None,
                    data.bytes,
                )
            })
            .collect();
        Writer::new(tensors)
            .write_to_file(path)
            .context("failed to write Muon state")?;
        Ok(())
    }

    /// Load Muon state from a byte buffer already authenticated by the
    /// checkpoint layer.
    pub fn load_bytes(&mut self, bytes: Vec<u8>, device: &Device) -> Result<()> {
        let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes))
            .context("failed to open authenticated Muon state")?;
        self.load_reader(reader, device)
    }

    fn load_reader(&mut self, reader: Reader, device: &Device) -> Result<()> {
        ensure!(
            reader.metadata().is_empty() && reader.scalars().is_empty(),
            "Muon checkpoint contains unsupported metadata or scalar state"
        );
        let mut velocities = BTreeMap::new();
        for tensor in reader.into_tensors().context("failed to read Muon state")? {
            ensure!(
                tensor.shape.rank() == 3,
                "Muon velocity {} has rank {}, expected 3",
                tensor.name,
                tensor.shape.rank()
            );
            let [batch, rows, columns] = tensor.shape.dims::<3>();
            ensure!(
                batch > 0 && rows > 0 && columns > 0,
                "Muon velocity {} has a zero dimension",
                tensor.name
            );
            ensure!(
                tensor.dtype == DType::F32,
                "Muon velocity {} has dtype {:?}, expected F32",
                tensor.name,
                tensor.dtype
            );
            ensure!(
                tensor.param_id.is_none(),
                "Muon velocity {} must not carry a parameter ID",
                tensor.name
            );
            ensure!(
                tensor.name == format!("{rows}x{columns}"),
                "Muon velocity {} does not match its canonical {rows}x{columns} shape name",
                tensor.name
            );
            let velocity = Tensor::<3>::from_data(
                TensorData::from_bytes(tensor.bytes, tensor.shape, tensor.dtype),
                device,
            );
            ensure!(
                velocities.insert([rows, columns], velocity).is_none(),
                "Muon checkpoint contains duplicate {rows}x{columns} velocity groups"
            );
        }
        self.velocities = velocities;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.velocities.is_empty()
    }

    /// Validate that every serialized velocity batch corresponds exactly to
    /// the selected matrix parameters in the restored model.
    pub fn validate_for_model<M: Module>(&self, model: &M, allow_empty: bool) -> Result<()> {
        let remaining = self
            .parameter_ids
            .iter()
            .map(|id| id.val())
            .collect::<BTreeSet<_>>();
        ensure!(
            remaining.len() == self.parameter_ids.len(),
            "Muon parameter selection repeats an ID"
        );
        let mut visitor = MuonShapeVisitor {
            remaining,
            groups: BTreeMap::new(),
            non_matrix: Vec::new(),
        };
        model.visit(&mut visitor);
        ensure!(
            visitor.remaining.is_empty(),
            "Muon selects {} parameters absent from the restored model",
            visitor.remaining.len()
        );
        ensure!(
            visitor.non_matrix.is_empty(),
            "Muon selects non-matrix parameters {:?}",
            visitor.non_matrix
        );
        if self.velocities.is_empty() {
            ensure!(
                allow_empty,
                "Muon checkpoint has no velocities after optimizer progress"
            );
            return Ok(());
        }
        ensure!(
            self.velocities.len() == visitor.groups.len(),
            "Muon checkpoint has {} velocity groups, restored model requires {}",
            self.velocities.len(),
            visitor.groups.len()
        );
        for (shape, expected_batch) in visitor.groups {
            let velocity = self.velocities.get(&shape).with_context(|| {
                format!(
                    "Muon checkpoint is missing the {}x{} velocity group",
                    shape[0], shape[1]
                )
            })?;
            let [actual_batch, rows, columns] = velocity.dims();
            ensure!(
                [rows, columns] == shape && actual_batch == expected_batch,
                "Muon {}x{} velocity batch has {actual_batch} matrices, restored model requires {expected_batch}",
                shape[0],
                shape[1]
            );
        }
        Ok(())
    }

    pub fn step<M: Module>(&mut self, lr: f64, model: M, mut grads: GradientsParams) -> Result<M> {
        let mut batches = BTreeMap::<[usize; 2], Vec<(ParamId, Tensor<2>)>>::new();
        for id in &self.parameter_ids {
            let grad = grads
                .remove::<2>(*id)
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
                updates.register::<2>(id, delta);
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

struct MuonShapeVisitor {
    remaining: BTreeSet<u64>,
    groups: BTreeMap<[usize; 2], usize>,
    non_matrix: Vec<u64>,
}

impl ModuleVisitor for MuonShapeVisitor {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        if !self.remaining.remove(&parameter.id.val()) {
            return;
        }
        if D != 2 {
            self.non_matrix.push(parameter.id.val());
            return;
        }
        let shape = parameter.shape().dims::<D>();
        *self.groups.entry([shape[0], shape[1]]).or_default() += 1;
    }
}

fn zeropower_via_newton_schulz(gradient: Tensor<3>) -> Tensor<3> {
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
fn to_compute_dtype(tensor: Tensor<3>) -> Tensor<3> {
    tensor.cast(FloatDType::BF16)
}

#[cfg(not(feature = "cuda"))]
fn to_compute_dtype(tensor: Tensor<3>) -> Tensor<3> {
    tensor
}

#[cfg(feature = "cuda")]
fn from_compute_dtype(tensor: Tensor<3>) -> Tensor<3> {
    tensor.cast(FloatDType::F32)
}

#[cfg(not(feature = "cuda"))]
fn from_compute_dtype(tensor: Tensor<3>) -> Tensor<3> {
    tensor
}

struct MuonUpdateMapper<'a> {
    updates: &'a mut GradientsParams,
}

impl ModuleMapper for MuonUpdateMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = match self.updates.remove::<D>(id) {
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
    use burn::tensor::TensorData;
    use burn_optim::MuonConfig;

    use super::*;

    fn packed_velocity(
        name: &str,
        shape: [usize; 3],
        dtype: DType,
        param_id: Option<u64>,
    ) -> Vec<u8> {
        let elements = shape.into_iter().product::<usize>();
        let tensor = PackedTensor::new(
            name.into(),
            dtype,
            shape.to_vec(),
            param_id,
            Bytes::from_bytes_vec(vec![0; elements * dtype.size()]),
        );
        Writer::new(vec![tensor]).into_bytes().unwrap().to_vec()
    }

    #[derive(Module, Debug)]
    struct MatrixPair {
        first: Param<Tensor<2>>,
        second: Param<Tensor<2>>,
    }

    impl MatrixPair {
        fn loss(&self, input: Tensor<2>) -> Tensor<1> {
            (input.clone().matmul(self.first.val()).square()
                + input.matmul(self.second.val()).square())
            .sum()
        }
    }

    fn values(model: &MatrixPair) -> Vec<f32> {
        [model.first.val(), model.second.val()]
            .into_iter()
            .flat_map(|tensor| tensor.into_data().to_vec::<f32>().unwrap())
            .collect()
    }

    #[test]
    fn batched_muon_matches_burn_for_repeated_shapes() {
        let device = hermes_llm::default_device();
        device.seed(17);
        let device = device.autodiff();
        let matrix = |scale: f32| {
            Param::from_tensor(
                Tensor::<2>::from_data(
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
            Tensor::<2>::from_data(
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
            expected = burn.step(2e-2.into(), expected, reference_grads);
        }

        let max_diff = values(&actual)
            .into_iter()
            .zip(values(&expected))
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 2e-5, "Muon parameter max diff: {max_diff}");
    }

    #[test]
    fn muon_archive_validation_is_strict_and_failure_atomic() {
        let device = hermes_llm::default_device();
        let mut muon = BatchedMuon::new(Vec::new());
        muon.velocities.insert(
            [2, 2],
            Tensor::<3>::ones([1, 2, 2], &device.clone().inner()),
        );
        let before = muon.velocities[&[2, 2]].clone().into_data();

        let error = muon
            .load_bytes(
                packed_velocity("not-a-shape", [1, 2, 2], DType::F32, None),
                &device.clone().inner(),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("canonical 2x2 shape name"));
        assert_eq!(
            muon.velocities[&[2, 2]].clone().into_data(),
            before,
            "a rejected archive changed existing Muon state"
        );

        for (bytes, message) in [
            (
                packed_velocity("2x2", [0, 2, 2], DType::F32, None),
                "zero dimension",
            ),
            (
                packed_velocity("2x2", [1, 2, 2], DType::F16, None),
                "expected F32",
            ),
            (
                packed_velocity("2x2", [1, 2, 2], DType::F32, Some(7)),
                "must not carry a parameter ID",
            ),
        ] {
            let error = BatchedMuon::new(Vec::new())
                .load_bytes(bytes, &device.clone().inner())
                .unwrap_err();
            assert!(format!("{error:#}").contains(message), "{error:#}");
        }
    }

    #[test]
    fn muon_velocity_batches_match_selected_model_shapes() {
        let device = hermes_llm::default_device();
        let matrix = || {
            Param::from_tensor(Tensor::<2>::zeros(
                [4, 6],
                &device.clone().autodiff().inner(),
            ))
        };
        let model = MatrixPair {
            first: matrix(),
            second: matrix(),
        };
        let duplicate = BatchedMuon::new(vec![model.first.id, model.first.id]);
        let error = duplicate.validate_for_model(&model, true).unwrap_err();
        assert!(format!("{error:#}").contains("repeats an ID"), "{error:#}");

        let ids = vec![model.first.id, model.second.id];
        let mut muon = BatchedMuon::new(ids);

        assert!(muon.validate_for_model(&model, true).is_ok());
        assert!(
            format!("{:#}", muon.validate_for_model(&model, false).unwrap_err())
                .contains("no velocities")
        );
        muon.velocities.insert(
            [4, 6],
            Tensor::<3>::zeros([1, 4, 6], &device.clone().inner()),
        );
        let error = muon.validate_for_model(&model, false).unwrap_err();
        assert!(format!("{error:#}").contains("requires 2"), "{error:#}");
        muon.velocities.insert(
            [4, 6],
            Tensor::<3>::zeros([2, 4, 6], &device.clone().inner()),
        );
        muon.validate_for_model(&model, false).unwrap();
    }
}
