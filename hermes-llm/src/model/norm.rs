//! Normalization selected by MAL `NormType`.
//!
//! RMSNorm and LayerNorm are Burn modules; `none` is an identity operation.

use burn::prelude::*;
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

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
            Self::Identity(_) => x,
        }
    }
}
