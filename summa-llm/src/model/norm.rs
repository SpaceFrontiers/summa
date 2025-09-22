//! Normalization selected by MAL `NormType`.
//!
//! RMSNorm and LayerNorm are Burn modules; `none` is an identity operation.

use burn::prelude::*;
use burn::tensor::DType;
use burn_nn::{LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig};

use crate::mal::NormType;

#[derive(Module, Debug)]
pub struct Identity {}

/// Unified normalization layer.
#[derive(Module, Debug)]
pub enum Norm {
    Rms(RmsNorm),
    Layer(LayerNorm),
    Identity(Identity),
}

impl Norm {
    pub fn new(norm_type: NormType, size: usize, eps: f64, device: &Device) -> Self {
        match norm_type {
            NormType::RmsNorm => Self::Rms(RmsNormConfig::new(size).with_epsilon(eps).init(device)),
            NormType::LayerNorm => {
                Self::Layer(LayerNormConfig::new(size).with_epsilon(eps).init(device))
            }
            NormType::None => Self::Identity(Identity {}),
        }
    }

    /// Normalization statistics always accumulate in FP32 (PyTorch autocast's
    /// layer-norm policy): a BF16 residual stream is cast up on entry and back
    /// on exit, and both casts fuse into the surrounding elementwise chains.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dtype = x.dtype();
        if matches!(self, Self::Identity(_)) || dtype == DType::F32 {
            return self.forward_f32(x);
        }
        self.forward_f32(x.cast(DType::F32)).cast(dtype)
    }

    fn forward_f32<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
            Self::Identity(_) => x,
        }
    }
}
