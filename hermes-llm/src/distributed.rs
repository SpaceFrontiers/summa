//! Distributed training support using NCCL
//!
//! This module provides multi-GPU training capabilities using NVIDIA's NCCL library
//! for efficient gradient synchronization across GPUs.

use anyhow::Result;
use candle_core::Tensor;

#[cfg(feature = "nccl")]
use cudarc::driver::safe::CudaDevice;
#[cfg(feature = "nccl")]
use cudarc::nccl::safe::{Comm, Id};
#[cfg(feature = "nccl")]
use std::rc::Rc;

/// Distributed configuration for multi-GPU training
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of GPUs/processes
    pub world_size: usize,
    /// This process's rank (0 to world_size-1)
    pub rank: usize,
    /// Path to communication file for NCCL ID exchange
    pub comm_file: String,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            comm_file: "nccl_id.txt".to_string(),
        }
    }
}

impl DistributedConfig {
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    pub fn is_main_process(&self) -> bool {
        self.rank == 0
    }
}

/// NCCL Communicator wrapper for gradient synchronization
#[cfg(feature = "nccl")]
pub struct NcclCommunicator {
    comm: Rc<Comm>,
    rank: usize,
    world_size: usize,
}

#[cfg(feature = "nccl")]
impl NcclCommunicator {
    /// Initialize NCCL communicator
    ///
    /// Rank 0 creates the NCCL ID and writes it to a file.
    /// Other ranks wait for the file and read the ID.
    pub fn new(config: &DistributedConfig) -> Result<Self> {
        use std::io::Write;

        let comm_file = std::path::PathBuf::from(&config.comm_file);

        // Rank 0 creates the ID, others wait for it
        let id = if config.rank == 0 {
            // Clean up any existing comm file
            if comm_file.exists() {
                std::fs::remove_file(&comm_file)?;
            }

            let id = Id::new().map_err(|e| anyhow::anyhow!("Failed to create NCCL ID: {:?}", e))?;

            // Write ID to temporary file then rename (atomic)
            let tmp_file = comm_file.with_extension("tmp");
            let mut file = std::fs::File::create(&tmp_file)?;
            file.write_all(&id.internal().iter().map(|&i| i as u8).collect::<Vec<_>>())?;
            std::fs::rename(&tmp_file, &comm_file)?;

            tracing::info!("Rank 0: Created NCCL ID and wrote to {:?}", comm_file);
            id
        } else {
            // Wait for rank 0 to create the file
            tracing::info!("Rank {}: Waiting for NCCL ID file...", config.rank);
            while !comm_file.exists() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            // Small delay to ensure file is fully written
            std::thread::sleep(std::time::Duration::from_millis(100));

            let data = std::fs::read(&comm_file)?;
            let internal: [i8; 128] = data
                .into_iter()
                .map(|i| i as i8)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Invalid NCCL ID file"))?;

            let id = Id::uninit(internal);
            tracing::info!("Rank {}: Read NCCL ID from {:?}", config.rank, comm_file);
            id
        };

        // Create CUDA device for this rank
        let cuda_device = CudaDevice::new(config.rank).map_err(|e| {
            anyhow::anyhow!("Failed to create CUDA device {}: {:?}", config.rank, e)
        })?;

        // Create NCCL communicator
        let comm = Comm::from_rank(cuda_device, config.rank, config.world_size, id)
            .map_err(|e| anyhow::anyhow!("Failed to create NCCL communicator: {:?}", e.0))?;

        // Rank 0 cleans up the comm file after all ranks have read it
        if config.rank == 0 {
            // Wait a bit for other ranks to read the file
            std::thread::sleep(std::time::Duration::from_secs(2));
            if comm_file.exists() {
                let _ = std::fs::remove_file(&comm_file);
            }
        }

        tracing::info!("Rank {}: NCCL communicator initialized", config.rank);

        Ok(Self {
            comm: Rc::new(comm),
            rank: config.rank,
            world_size: config.world_size,
        })
    }

    /// All-reduce a tensor (sum across all ranks, then divide by world_size for average)
    pub fn all_reduce_avg(&self, tensor: &Tensor) -> Result<Tensor> {
        // For now, we use the synchronous all-reduce
        // The tensor must be on a CUDA device
        let reduced = self.all_reduce_sum(tensor)?;
        let avg = reduced.affine(1.0 / self.world_size as f64, 0.0)?;
        Ok(avg)
    }

    /// All-reduce a tensor (sum across all ranks)
    pub fn all_reduce_sum(&self, tensor: &Tensor) -> Result<Tensor> {
        use cudarc::nccl::safe::ReduceOp;

        // Get the underlying CUDA storage
        let storage = tensor.storage_and_layout().0;

        // Perform all-reduce
        // Note: This is a simplified implementation
        // In practice, you'd need to handle the tensor's CUDA buffer directly

        // For now, we'll use a workaround by converting to/from vec
        // This is not optimal but works for demonstration
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        // Create output buffer
        let mut output = data.clone();

        // Perform NCCL all-reduce
        // Note: actual implementation would use comm.all_reduce directly on GPU buffers
        self.comm
            .all_reduce(&data, &mut output, &ReduceOp::Sum)
            .map_err(|e| anyhow::anyhow!("NCCL all-reduce failed: {:?}", e.0))?;

        // Convert back to tensor
        let result = Tensor::from_vec(output, tensor.shape(), tensor.device())?;
        Ok(result)
    }

    /// Broadcast a tensor from rank 0 to all other ranks
    pub fn broadcast(&self, tensor: &Tensor) -> Result<Tensor> {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let mut output = data.clone();

        self.comm
            .broadcast(&data, &mut output, 0)
            .map_err(|e| anyhow::anyhow!("NCCL broadcast failed: {:?}", e.0))?;

        let result = Tensor::from_vec(output, tensor.shape(), tensor.device())?;
        Ok(result)
    }

    /// Synchronize all ranks (barrier)
    pub fn barrier(&self) -> Result<()> {
        // Use a small all-reduce as a barrier
        let dummy = vec![0.0f32];
        let mut output = dummy.clone();
        self.comm
            .all_reduce(&dummy, &mut output, &cudarc::nccl::safe::ReduceOp::Sum)
            .map_err(|e| anyhow::anyhow!("NCCL barrier failed: {:?}", e.0))?;
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Stub communicator for non-NCCL builds
#[cfg(not(feature = "nccl"))]
pub struct NcclCommunicator {
    rank: usize,
    world_size: usize,
}

#[cfg(not(feature = "nccl"))]
impl NcclCommunicator {
    pub fn new(_config: &DistributedConfig) -> Result<Self> {
        anyhow::bail!("NCCL support not enabled. Build with --features nccl")
    }

    pub fn all_reduce_avg(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    pub fn all_reduce_sum(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    pub fn broadcast(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    pub fn barrier(&self) -> Result<()> {
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Synchronize gradients across all ranks using NCCL
pub fn sync_gradients(var_map: &candle_nn::VarMap, comm: &NcclCommunicator) -> Result<()> {
    for var in var_map.all_vars() {
        let tensor = var.as_tensor();
        let synced = comm.all_reduce_avg(tensor)?;
        var.set(&synced)?;
    }
    Ok(())
}
