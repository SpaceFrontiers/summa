//! Compile-time inference backend selection.
//!
//! One compute stack, three targets, chosen at compile time by cargo feature:
//! - `metal`   → wgpu (Metal on macOS, Vulkan/DX elsewhere)
//! - `cuda`    → CUDA
//! - neither             → ndarray (CPU; also the test/parity backend)

/// The active Burn backend for inference. Inference-only — never wrapped in
/// `Autodiff` (no training graph, so no autodiff overhead).
#[cfg(feature = "metal")]
pub type Backend = burn_wgpu::Wgpu;

#[cfg(all(feature = "cuda", not(feature = "metal")))]
pub type Backend = burn_cuda::Cuda;

#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub type Backend = burn_ndarray::NdArray;

/// Device handle for the active backend.
pub type Device = burn::tensor::Device<Backend>;

/// The default device for the active backend (best-available GPU, or CPU).
pub fn default_device() -> Device {
    Device::default()
}
