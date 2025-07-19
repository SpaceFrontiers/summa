//! Fast top-2 router selection.
//!
//! Generic `topk` sorts more values than an MoE router needs. GPU backends use
//! one thread per token to scan the small expert axis once; selected logits are
//! gathered by ordinary tensor operations so router gradients remain visible
//! to autodiff. Other `top_k` values keep Burn's general implementation.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
use burn::backend::NdArray;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::*;
use burn::tensor::Int;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;

#[cfg_attr(feature = "cuda", backend_extension(Cuda, Autodiff))]
#[cfg_attr(
    all(not(feature = "cuda"), feature = "metal"),
    backend_extension(Metal, Autodiff)
)]
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal")),
    backend_extension(NdArray, Autodiff)
)]
pub trait MoeTop2Backend: Backend {
    fn moe_top2_indices(logits: FloatTensor<Self>) -> IntTensor<Self>;
}

pub(super) fn top2_indices(logits: Tensor<2>) -> Tensor<2, Int> {
    if logits.device() == Device::ndarray() {
        return logits.topk_with_indices(2, 1).1;
    }
    Tensor::from_dispatch(Dispatch::moe_top2_indices(logits.into_dispatch()))
}

impl MoeTop2Backend for burn_ndarray::NdArray {
    fn moe_top2_indices(logits: FloatTensor<Self>) -> IntTensor<Self> {
        Tensor::<2>::from_primitive::<Self>(logits)
            .topk_with_indices(2, 1)
            .1
            .try_into_primitive::<Self>()
            .expect("top-2 indices stayed on the ndarray backend")
    }
}

impl<B: MoeTop2Backend, C: CheckpointStrategy> MoeTop2Backend for Autodiff<B, C> {
    fn moe_top2_indices(logits: FloatTensor<Self>) -> IntTensor<Self> {
        <Self as AutodiffBackend>::int_from_inner(B::moe_top2_indices(logits.primitive))
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};
    use half::bf16;

    use super::MoeTop2Backend;
    use crate::model::cube_tensor::{empty_like_dtype, into_contiguous};

    const THREADS: u32 = 256;

    #[cube(launch)]
    fn top2_indices_kernel<F: Float>(
        logits: &Tensor<F>,
        output: &mut Tensor<i32>,
        experts: u32,
        tokens: u32,
    ) {
        let token = ABSOLUTE_POS;
        let tokens = tokens as usize;
        if token < tokens {
            let experts = experts as usize;
            let base = token * experts;
            let mut first_value = f32::cast_from(f32::NEG_INFINITY);
            let mut second_value = f32::cast_from(f32::NEG_INFINITY);
            let mut first_index = 0u32;
            let mut second_index = 0u32;
            for expert in 0..experts {
                let value = f32::cast_from(logits[base + expert]);
                if value > first_value {
                    second_value = first_value;
                    second_index = first_index;
                    first_value = value;
                    first_index = expert as u32;
                } else if value > second_value {
                    second_value = value;
                    second_index = expert as u32;
                }
            }
            output[token * 2] = i32::cast_from(first_index);
            output[token * 2 + 1] = i32::cast_from(second_index);
        }
    }

    impl<R: CubeRuntime> MoeTop2Backend for CubeBackend<R> {
        fn moe_top2_indices(logits: CubeTensor<R>) -> CubeTensor<R> {
            let [tokens, experts] = logits.meta.shape.dims();
            assert!(experts >= 2, "top-2 routing needs at least two experts");
            let logits = into_contiguous(logits);
            let output = empty_like_dtype(&logits, Shape::new([tokens, 2]), DType::I32);
            let client = logits.client.clone();
            macro_rules! launch {
                ($float:ty) => {
                    top2_indices_kernel::launch::<$float, R>(
                        &client,
                        CubeCount::Static((tokens as u32).div_ceil(THREADS), 1, 1),
                        CubeDim::new_1d(THREADS),
                        logits.into_tensor_arg(),
                        output.clone().into_tensor_arg(),
                        experts as u32,
                        tokens as u32,
                    )
                };
            }
            match logits.dtype {
                DType::BF16 => launch!(bf16),
                DType::F32 => launch!(f32),
                dtype => panic!("MoE router logits must be F32 or BF16, got {dtype:?}"),
            }
            output
        }
    }
}

#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::top2_indices;

    #[test]
    fn gpu_top2_matches_general_cpu_topk() {
        let values = vec![
            0.2, -0.7, 1.1, 0.5, 0.4, // 2, 3
            -0.2, 2.0, 0.1, 1.7, 0.9, // 1, 3
            3.0, 2.9, -4.0, 0.0, 1.0, // 0, 1
            -3.0, -0.1, -0.2, -0.4, -0.3, // 1, 2
        ];
        let cpu =
            Tensor::<2>::from_data(TensorData::new(values.clone(), [4, 5]), &Device::ndarray());
        let gpu = Tensor::<2>::from_data(
            TensorData::new(values, [4, 5]),
            &crate::model::default_device(),
        );
        let expected = top2_indices(cpu).into_data().convert::<i64>();
        let actual = top2_indices(gpu).into_data().convert::<i64>();
        assert_eq!(actual, expected);
    }
}
