use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::mal::NormType;

#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(dtype)?;
        x.broadcast_mul(&self.weight)
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize, eps: f64, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        let bias = if use_bias {
            Some(vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?)
        } else {
            None
        };
        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_sub(&mean)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(dtype)?;
        let x = x.broadcast_mul(&self.weight)?;
        match &self.bias {
            Some(bias) => x.broadcast_add(bias),
            None => Ok(x),
        }
    }
}

/// Unified normalization layer that can be either RMSNorm or LayerNorm
#[derive(Debug, Clone)]
pub enum Norm {
    RmsNorm(RMSNorm),
    LayerNorm(LayerNorm),
}

impl Norm {
    pub fn new(
        norm_type: NormType,
        size: usize,
        eps: f64,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        match norm_type {
            NormType::RmsNorm | NormType::None => Ok(Self::RmsNorm(RMSNorm::new(size, eps, vb)?)),
            NormType::LayerNorm => Ok(Self::LayerNorm(LayerNorm::new(size, eps, use_bias, vb)?)),
        }
    }
}

impl Module for Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::RmsNorm(n) => n.forward(x),
            Self::LayerNorm(n) => n.forward(x),
        }
    }
}
