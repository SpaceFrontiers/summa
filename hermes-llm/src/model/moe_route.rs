//! GPU-resident stable MoE route planning.
//!
//! Only the small per-expert count vector is read by the host to define the
//! grouped-GEMM shapes. Route sorting and the inverse permutation stay on the
//! accelerator, avoiding a full route-index download and two uploads.

use burn::backend::Cuda;
use burn::backend::{AutodiffBackend, Backend, Dispatch, backend_extension, tensor::IntTensor};
use burn::prelude::*;
use burn::tensor::Int;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::CheckpointStrategy;

#[backend_extension(Cuda, Autodiff)]
pub trait MoeRouteBackend: Backend {
    fn moe_route_plan(
        indices: IntTensor<Self>,
        expert_count: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>);
}

pub(super) fn route_plan(
    indices: Tensor<2, Int>,
    expert_count: usize,
) -> (Tensor<1, Int>, Tensor<1, Int>, Tensor<1, Int>) {
    let (order, inverse, counts) = Dispatch::moe_route_plan(indices.into_dispatch(), expert_count);
    (
        Tensor::from_dispatch(order),
        Tensor::from_dispatch(inverse),
        Tensor::from_dispatch(counts),
    )
}

impl<B: MoeRouteBackend, C: CheckpointStrategy> MoeRouteBackend for Autodiff<B, C> {
    fn moe_route_plan(
        indices: IntTensor<Self>,
        expert_count: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>) {
        let indices = <Self as AutodiffBackend>::int_inner(indices);
        let (order, inverse, counts) = B::moe_route_plan(indices, expert_count);
        (
            <Self as AutodiffBackend>::int_from_inner(order),
            <Self as AutodiffBackend>::int_from_inner(inverse),
            <Self as AutodiffBackend>::int_from_inner(counts),
        )
    }
}

mod gpu {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::tensor::CubeTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime};

    use super::MoeRouteBackend;
    use crate::model::cube_tensor::{empty_like_dtype, into_contiguous};

    const ROUTE_THREADS: u32 = 256;

    #[cube(launch)]
    fn route_counts(
        indices: &Tensor<i32>,
        counts_out: &mut Tensor<i32>,
        routes: u32,
        #[comptime] expert_count: u32,
    ) {
        let lane = UNIT_POS_X;
        let counts = Shared::<[Atomic<i32>]>::new_slice(expert_count as usize);
        if lane < expert_count {
            counts[lane as usize].store(0i32);
        }
        sync_cube();

        let mut route = lane;
        while route < routes {
            let expert = indices[route as usize];
            counts[expert as usize].fetch_add(1i32);
            route += ROUTE_THREADS;
        }
        sync_cube();

        if lane < expert_count {
            counts_out[lane as usize] = counts[lane as usize].load();
        }
    }

    #[cube(launch)]
    fn stable_route_pack(
        indices: &Tensor<i32>,
        counts: &Tensor<i32>,
        order: &mut Tensor<i32>,
        inverse: &mut Tensor<i32>,
        routes: u32,
        #[comptime] expert_count: u32,
    ) {
        let expert = CUBE_POS_X;
        let lane = UNIT_POS_X;
        if expert < expert_count {
            let mut expert_offset = 0i32;
            let mut previous = 0u32;
            while previous < expert {
                expert_offset += counts[previous as usize];
                previous += 1;
            }

            let mut flags = Shared::new_slice(ROUTE_THREADS as usize);
            let mut chunk_start = 0u32;
            let mut written = 0i32;
            while chunk_start < routes {
                let route = chunk_start + lane;
                let selected = route < routes && indices[route as usize] == expert as i32;
                flags[lane as usize] = if selected { 1i32 } else { 0i32 };
                sync_cube();

                // Stable inclusive scan of this route chunk. Every thread
                // participates in every barrier, including tail chunks.
                let mut distance = 1u32;
                while distance < ROUTE_THREADS {
                    let add = if lane >= distance {
                        flags[(lane - distance) as usize]
                    } else {
                        0i32
                    };
                    sync_cube();
                    if lane >= distance {
                        flags[lane as usize] += add;
                    }
                    sync_cube();
                    distance *= 2;
                }

                let chunk_count = flags[(ROUTE_THREADS - 1) as usize];
                if selected {
                    let sorted = expert_offset + written + flags[lane as usize] - 1;
                    order[sorted as usize] = route as i32;
                    inverse[route as usize] = sorted;
                }
                written += chunk_count;
                sync_cube();
                chunk_start += ROUTE_THREADS;
            }
        }
    }

    impl<R: CubeRuntime> MoeRouteBackend for CubeBackend<R> {
        fn moe_route_plan(
            indices: CubeTensor<R>,
            expert_count: usize,
        ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
            let indices = into_contiguous(indices);
            let [tokens, top_k] = indices.meta.shape.dims();
            let routes = tokens.checked_mul(top_k).expect("MoE route count overflow");
            assert!(routes > 0, "MoE route planning requires at least one route");
            assert!(
                (1..=ROUTE_THREADS as usize).contains(&expert_count),
                "GPU MoE routing supports 1..={ROUTE_THREADS} experts"
            );
            assert_eq!(indices.dtype, DType::I32, "MoE indices must use i32");
            let order = empty_like_dtype(&indices, Shape::new([routes]), DType::I32);
            let inverse = empty_like_dtype(&indices, Shape::new([routes]), DType::I32);
            let counts = empty_like_dtype(&indices, Shape::new([expert_count]), DType::I32);
            let client = indices.client.clone();

            route_counts::launch::<R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(ROUTE_THREADS),
                indices.clone().into_tensor_arg(),
                counts.clone().into_tensor_arg(),
                routes as u32,
                expert_count as u32,
            );
            stable_route_pack::launch::<R>(
                &client,
                CubeCount::Static(expert_count as u32, 1, 1),
                CubeDim::new_1d(ROUTE_THREADS),
                indices.into_tensor_arg(),
                counts.clone().into_tensor_arg(),
                order.clone().into_tensor_arg(),
                inverse.clone().into_tensor_arg(),
                routes as u32,
                expert_count as u32,
            );
            (order, inverse, counts)
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    #[test]
    fn gpu_route_plan_is_stable_and_invertible() {
        let device = crate::model::default_device();
        let indices = Tensor::<2, Int>::from_data(
            TensorData::new(vec![2_i64, 0, 1, 2, 0, 1, 2, 1], [4, 2]),
            &device,
        );
        let (order, inverse, counts) = route_plan(indices, 3);
        assert_eq!(
            counts.into_data().convert::<i64>(),
            TensorData::new(vec![2_i64, 3, 3], [3])
        );
        assert_eq!(
            order.clone().into_data().convert::<i64>(),
            TensorData::new(vec![1_i64, 4, 2, 5, 7, 0, 3, 6], [8])
        );
        let restored = order.select(0, inverse);
        assert_eq!(
            restored.into_data().convert::<i64>(),
            TensorData::new((0_i64..8).collect::<Vec<_>>(), [8])
        );
    }
}
