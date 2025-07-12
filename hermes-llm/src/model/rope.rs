use candle_core::{Device, Result, Tensor};

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// `scaling` is linear position interpolation (NTK-free): positions are
    /// divided by the factor, extending the effective context. `None`/`<=1`
    /// means no scaling.
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        scaling: Option<f64>,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let scale = match scaling {
            Some(s) if s > 1.0 => s as f32,
            _ => 1.0,
        };
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32 / scale).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let q_rot = self.rotate_half(q, &cos, &sin)?;
        let k_rot = self.rotate_half(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, h, seq, d) = x.dims4()?;
        let x1 = x.narrow(3, 0, d / 2)?;
        let x2 = x.narrow(3, d / 2, d / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

        let cos = cos
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, h, seq, d / 2))?;
        let sin = sin
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, h, seq, d / 2))?;
        let cos = Tensor::cat(&[&cos, &cos], 3)?;
        let sin = Tensor::cat(&[&sin, &sin], 3)?;

        let x_cos = x.broadcast_mul(&cos)?;
        let rot_sin = rotated.broadcast_mul(&sin)?;
        let result = x_cos.add(&rot_sin)?;
        Ok(result)
    }
}
