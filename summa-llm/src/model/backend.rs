//! Runtime tensor-device selection.

pub use burn::tensor::Device;

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn configure_cuda_memory() {
    use std::sync::Once;

    use burn_cubecl::cubecl::{
        Runtime,
        config::{
            memory::{MemoryPoolConfig, MemoryPoolsConfig},
            size::MemorySize,
        },
        cuda::{CudaDevice, CudaRuntime},
    };

    static CONFIGURE: Once = Once::new();
    CONFIGURE.call_once(|| {
        let client = CudaRuntime::client(&CudaDevice::default());
        // Isolate allocations in power-of-two pages. The default sliced pools
        // reserve multi-gigabyte backing pages for the largest training tensors;
        // exclusive pages avoid that amplification, while CubeCL's allocation
        // retry reclaims unused autotune pages only when memory is pressured.
        let pools = (15..=34) // 32 KiB through 16 GiB.
            .map(|shift| MemoryPoolConfig::Exclusive {
                max_alloc_size: MemorySize(1_u64 << shift),
                dealloc_period: None,
            })
            .collect();
        assert!(
            client.configure_memory_pools(&MemoryPoolsConfig::Explicit(pools)),
            "CUDA memory pools must be configured before allocating tensors"
        );
    });
}

/// The default device for the active backend (best-available GPU, or CPU).
pub fn default_device() -> Device {
    #[cfg(feature = "metal")]
    {
        Device::metal(burn::tensor::DeviceKind::DefaultDevice)
    }

    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    {
        configure_cuda_memory();
        Device::cuda(burn::tensor::DeviceIndex::Default)
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        Device::ndarray()
    }
}
