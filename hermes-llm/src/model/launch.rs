//! Legal factoring of 1-D kernel launches.
//!
//! wgpu (and therefore Metal) caps every dispatch dimension at 65,535
//! workgroups, so a linear cube count like `elements.div_ceil(THREADS)`
//! becomes an invalid dispatch for large inputs (a 16k-token MoE batch is
//! enough). CUDA only caps the y/z axes, which is why the same launches
//! worked there. The factoring below keeps every dimension within the
//! wgpu limit on all backends and is a no-op (identical 1-D shape) below
//! 65,535 groups.
//!
//! Kernels launched with a factored count must index by `ABSOLUTE_POS`
//! (or `CUBE_POS` for one-cube-per-row kernels) and bounds-check against
//! their element count: CubeCL linearizes both builtins across every
//! launch dimension, and the factoring may overshoot by up to one row of
//! excess cubes.

/// Maximum workgroups per dispatch dimension on wgpu/Metal. CUDA allows
/// 2^31-1 along x, but counts within this bound are legal everywhere, so
/// one factoring serves all backends.
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
pub(crate) const MAX_CUBES_PER_DIM: u32 = 65_535;

/// Factors a required workgroup count into `(x, y, z)` with every
/// dimension `<= MAX_CUBES_PER_DIM` and `x * y * z >= groups`, keeping
/// the overshoot minimal (less than one y-row of cubes). Allocation-free.
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
pub(crate) fn linear_launch_dims(groups: u32) -> (u32, u32, u32) {
    if groups <= MAX_CUBES_PER_DIM {
        return (groups, 1, 1);
    }
    let y = groups.div_ceil(MAX_CUBES_PER_DIM);
    if y <= MAX_CUBES_PER_DIM {
        // x = ceil(groups / y) <= MAX because y >= groups / MAX.
        return (groups.div_ceil(y), y, 1);
    }
    // Only reachable for groups > 65,535^2: the div_ceil overshoot pushes
    // y past the limit near u32::MAX, where z is at most 2.
    let z = y.div_ceil(MAX_CUBES_PER_DIM);
    let plane = groups.div_ceil(z);
    let y = plane.div_ceil(MAX_CUBES_PER_DIM);
    (plane.div_ceil(y), y, z)
}

/// A legal `CubeCount` for `groups` linearly indexed workgroups; see
/// [`linear_launch_dims`] for the kernel-side contract.
#[cfg(any(feature = "metal", feature = "cuda"))]
pub(crate) fn linear_cube_count(groups: u32) -> burn_cubecl::cubecl::prelude::CubeCount {
    let (x, y, z) = linear_launch_dims(groups);
    burn_cubecl::cubecl::prelude::CubeCount::Static(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::{MAX_CUBES_PER_DIM, linear_launch_dims};

    /// Checks the helper's full contract for one input: every dimension is
    /// within the wgpu limit, the grid covers `groups`, and the overshoot
    /// stays below one y-row of cubes per z-plane.
    fn assert_legal(groups: u32) -> (u32, u32, u32) {
        let (x, y, z) = linear_launch_dims(groups);
        assert!(
            x <= MAX_CUBES_PER_DIM,
            "{groups}: x = {x} exceeds the limit"
        );
        assert!(
            y <= MAX_CUBES_PER_DIM,
            "{groups}: y = {y} exceeds the limit"
        );
        assert!(
            z <= MAX_CUBES_PER_DIM,
            "{groups}: z = {z} exceeds the limit"
        );
        let capacity = u64::from(x) * u64::from(y) * u64::from(z);
        assert!(
            capacity >= u64::from(groups),
            "{groups}: capacity {capacity} does not cover the request"
        );
        assert!(
            capacity - u64::from(groups) < u64::from(z) * (u64::from(y) + 1),
            "{groups}: overshoot {} is not minimal for ({x}, {y}, {z})",
            capacity - u64::from(groups)
        );
        (x, y, z)
    }

    #[test]
    fn small_counts_stay_one_dimensional() {
        assert_eq!(assert_legal(0), (0, 1, 1));
        assert_eq!(assert_legal(1), (1, 1, 1));
        assert_eq!(assert_legal(256), (256, 1, 1));
        assert_eq!(assert_legal(MAX_CUBES_PER_DIM), (MAX_CUBES_PER_DIM, 1, 1));
    }

    #[test]
    fn first_illegal_count_splits_exactly() {
        // 65,536 = 32,768 * 2 with zero overshoot.
        assert_eq!(assert_legal(MAX_CUBES_PER_DIM + 1), (32_768, 2, 1));
    }

    #[test]
    fn production_shapes_are_exact_or_near_exact() {
        // The observed Metal failure: 16k-token MoE batch, 33.5M elements
        // at 256 threads.
        assert_legal(131_072);
        // 8k tokens x 1024 channels at 128 threads (softplus).
        assert_legal(65_536);
    }

    #[test]
    fn large_primes_overshoot_minimally() {
        for prime in [2_147_483_647_u32, 4_294_967_291, 179_424_691, 87_178_291] {
            assert_legal(prime);
        }
    }

    #[test]
    fn two_dimensional_boundary_and_u32_extremes_use_z() {
        let squared = MAX_CUBES_PER_DIM * MAX_CUBES_PER_DIM; // 4,294,836,225
        assert_eq!(
            assert_legal(squared),
            (MAX_CUBES_PER_DIM, MAX_CUBES_PER_DIM, 1)
        );
        let (_, _, z) = assert_legal(squared + 1);
        assert!(z >= 2, "counts beyond 65,535^2 must spill into z");
        let (_, _, z) = assert_legal(u32::MAX);
        assert_eq!(z, 2);
    }
}
