//! Runtime tensor-device selection.

pub use burn::tensor::Device;

/// The default device for the active backend (best-available GPU, or CPU).
pub fn default_device() -> Device {
    #[cfg(feature = "metal")]
    return Device::metal(burn::tensor::DeviceKind::DefaultDevice);

    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    return Device::cuda(burn::tensor::DeviceIndex::Default);

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    return Device::ndarray();
}
