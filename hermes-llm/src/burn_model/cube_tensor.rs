//! Shared allocation helpers for custom CubeCL operators.

use burn::tensor::Shape;
use burn_cubecl::CubeRuntime;
use burn_cubecl::tensor::CubeTensor;

pub(super) fn into_contiguous<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    burn_cubecl::kernel::into_contiguous(tensor)
}

pub(super) fn empty_like<R: CubeRuntime>(tensor: &CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::empty_device_contiguous_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape,
        tensor.dtype,
    )
}

pub(super) fn zeros_like<R: CubeRuntime>(tensor: &CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::zeros_client(
        tensor.client.clone(),
        tensor.device.clone(),
        shape,
        tensor.dtype,
    )
}
