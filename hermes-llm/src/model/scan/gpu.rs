//! CubeCL kernels and dispatch for the selective scan.
//!
//! One swept, checkpointed kernel per direction carries the training path
//! for every batch and sequence length: a block owns a `(batch, channel
//! tile)` pair and walks the checkpoint segments with the running state
//! (forward) or adjoint (backward) in registers, staging each segment's
//! sequence tiles through workgroup memory once and folding softplus into
//! the kernels. Cross-thread reductions land in disjoint per-warp slots
//! flushed every `BACKWARD_FLUSH` steps. The remaining kernels serve
//! inference (decode step, prefill with and without state-lane
//! parallelism).
//!
//! The warp geometry (a plane spans two state rows of a
//! `BACKWARD_CHANNELS`-wide channel tile) is compile-time asserted, and
//! the dispatch enforces the measured power-of-two state-width contract —
//! see `docs/kernel-tuning-surface.md`.

use burn::backend::TensorMetadata;
use burn::tensor::Shape;
use burn_cubecl::cubecl::ir::FastMath;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime};

use burn::tensor::DType;
use half::bf16;

use super::{CHECKPOINTED_SCAN_INTERVAL, MambaBackend};
use crate::model::cube_tensor::{empty_like, empty_like_dtype, into_contiguous, zeros_like_dtype};

const THREADS_PER_CUBE: u32 = 128;
const PLANE_WIDTH: u32 = 32;
const FORWARD_CHANNELS: u32 = THREADS_PER_CUBE / PLANE_WIDTH;
// A100 measurements favor serial recurrence once this many independent
// blocks are available; smaller grids benefit from parallel state lanes.
const SERIAL_SCAN_MIN_BLOCKS: u32 = 128;
const BACKWARD_CHANNELS: usize = 16;
// Reverse-sweep steps buffered between flush barriers in the segmented
// backward. Sized so the per-warp partial slots fit Metal's 32KB
// threadgroup budget at state_dim 16; must divide the segment length.
const BACKWARD_FLUSH: usize = 8;
// The sweep kernels' warp geometry: one plane spans exactly two state
// rows of a BACKWARD_CHANNELS-wide channel tile, so the partner-row
// shuffle offset is BACKWARD_CHANNELS and `half_plane_sum` reduces
// BACKWARD_CHANNELS-lane groups. Changing the tile width means
// revisiting both, and the flush windows must tile the segment.
const _: () = assert!(BACKWARD_CHANNELS * 2 == PLANE_WIDTH as usize);
const _: () = assert!(CHECKPOINTED_SCAN_INTERVAL.is_multiple_of(BACKWARD_FLUSH));
// The dispatch caps state_dim at 16: 32 works on CUDA but exceeds
// Metal's 32KB threadgroup budget in the reverse sweep, and the CUDA
// runtime displaces writes into state tensors whose minor stride is
// not a power of two (4/8/16/32 measured clean, 2/5/6/12 corrupt) —
// the kernels themselves are shape-generic, as Metal demonstrates.

#[cube]
fn atomic_add_f32(target: &mut Atomic<f32>, value: f32) {
    target.fetch_add(value);
}

#[cube]
fn stable_softplus(value: f32) -> f32 {
    if value > 0.0 {
        value + ((-value).exp() + 1.0).ln()
    } else {
        (value.exp() + 1.0).ln()
    }
}

#[cube]
fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

/// Folds a BACKWARD_CHANNELS-lane group (half a plane) down to its
/// lane 0. Both halves of a plane reduce independently, which is what
/// the sweep kernels' two-state-rows-per-warp layout needs.
#[cube]
fn half_plane_sum(value: f32, lane: usize) -> f32 {
    let mut sum = value;
    let other = plane_shuffle_down(sum, (BACKWARD_CHANNELS / 2) as u32);
    if lane < BACKWARD_CHANNELS / 2 {
        sum += other;
    }
    let other = plane_shuffle_down(sum, (BACKWARD_CHANNELS / 4) as u32);
    if lane < BACKWARD_CHANNELS / 4 {
        sum += other;
    }
    let other = plane_shuffle_down(sum, (BACKWARD_CHANNELS / 8) as u32);
    if lane < BACKWARD_CHANNELS / 8 {
        sum += other;
    }
    let other = plane_shuffle_down(sum, (BACKWARD_CHANNELS / 16) as u32);
    if lane < BACKWARD_CHANNELS / 16 {
        sum += other;
    }
    sum
}
// The reduction ladder above is written for a 16-wide tile.
const _: () = assert!(BACKWARD_CHANNELS == 16);

#[cube(launch)]
fn softplus_forward<F: Float>(input: &Tensor<F>, output: &mut Tensor<f32>) {
    let idx = ABSOLUTE_POS;
    if idx < input.len() {
        output[idx] = stable_softplus(f32::cast_from(input[idx]));
    }
}

#[cube(launch)]
fn selective_scan_step(
    delta: &Tensor<f32>,
    xs: &Tensor<f32>,
    b_mat: &Tensor<f32>,
    c_mat: &Tensor<f32>,
    a: &Tensor<f32>,
    d: &Tensor<f32>,
    h_in: &Tensor<f32>,
    y: &mut Tensor<f32>,
    h_out: &mut Tensor<f32>,
    channels: u32,
    #[comptime] state_dim: usize,
) {
    let channels = channels as usize;
    let idx = ABSOLUTE_POS;
    let total = xs.len();
    if idx < total {
        let batch = idx / channels;
        let channel = idx % channels;
        let state_base = idx * state_dim;
        let a_base = channel * state_dim;
        let mut state = Array::<f32>::new(state_dim);
        for n in 0..state_dim {
            state[n] = h_in[state_base + n];
        }

        let btn = batch * state_dim;
        let dt = delta[idx];
        let x = xs[idx];
        let mut out = 0.0f32;
        for n in 0..state_dim {
            let da = (dt * a[a_base + n]).exp();
            state[n] = state[n] * da + dt * b_mat[btn + n] * x;
            out += state[n] * c_mat[btn + n];
            h_out[state_base + n] = state[n];
        }
        y[idx] = out + x * d[channel];
    }
}

/// One thread owns a batch/channel pair. This avoids a plane reduction
/// when the batch/channel grid already has enough blocks to fill the GPU.
#[allow(clippy::manual_div_ceil)]
#[cube(launch)]
fn selective_scan_forward_serial(
    delta: &Tensor<f32>,
    xs: &Tensor<f32>,
    b_mat: &Tensor<f32>,
    c_mat: &Tensor<f32>,
    a: &Tensor<f32>,
    d: &Tensor<f32>,
    h_in: &Tensor<f32>,
    y: &mut Tensor<f32>,
    checkpoints: &mut Tensor<f32>,
    h_out: &mut Tensor<f32>,
    channels: u32,
    seq_len: u32,
    #[comptime] state_dim: usize,
    #[comptime] checkpoint_interval: usize,
    #[comptime] save_checkpoints: bool,
) {
    let channels = channels as usize;
    let seq_len = seq_len as usize;
    let idx = ABSOLUTE_POS;
    let batch_channels = xs.len() / seq_len;
    if idx < batch_channels {
        let batch = idx / channels;
        let channel = idx % channels;
        let state_base = idx * state_dim;
        let a_base = channel * state_dim;
        let checkpoint_count = (seq_len + checkpoint_interval - 1) / checkpoint_interval;
        let mut state = Array::<f32>::new(state_dim);
        for n in 0..state_dim {
            state[n] = h_in[state_base + n];
        }

        for t in 0..seq_len {
            let btc = (batch * seq_len + t) * channels + channel;
            let btn = (batch * seq_len + t) * state_dim;
            let dt = delta[btc];
            let x = xs[btc];
            let mut out = 0.0f32;
            for n in 0..state_dim {
                state[n] = state[n] * (dt * a[a_base + n]).exp() + dt * b_mat[btn + n] * x;
                out += state[n] * c_mat[btn + n];
            }
            y[btc] = out + x * d[channel];

            if save_checkpoints && ((t + 1) % checkpoint_interval == 0 || t + 1 == seq_len) {
                let checkpoint = t / checkpoint_interval;
                let checkpoint_base =
                    ((batch * checkpoint_count + checkpoint) * channels + channel) * state_dim;
                for n in 0..state_dim {
                    checkpoints[checkpoint_base + n] = state[n];
                }
            }
        }
        for n in 0..state_dim {
            h_out[state_base + n] = state[n];
        }
    }
}

/// One plane owns a batch/channel pair. This exposes the state dimension
/// as parallel work when a small batch would otherwise underfill the GPU.
#[allow(clippy::manual_div_ceil)]
#[cube(launch)]
fn selective_scan_forward_parallel(
    delta: &Tensor<f32>,
    xs: &Tensor<f32>,
    b_mat: &Tensor<f32>,
    c_mat: &Tensor<f32>,
    a: &Tensor<f32>,
    d: &Tensor<f32>,
    h_in: &Tensor<f32>,
    y: &mut Tensor<f32>,
    checkpoints: &mut Tensor<f32>,
    h_out: &mut Tensor<f32>,
    channels: u32,
    seq_len: u32,
    #[comptime] state_dim: usize,
    #[comptime] checkpoint_interval: usize,
    #[comptime] save_checkpoints: bool,
) {
    let channels = channels as usize;
    let seq_len = seq_len as usize;
    let n = UNIT_POS_X as usize;
    let local_channel = UNIT_POS_Y as usize;
    let batch = CUBE_POS_Y as usize;
    let channel = CUBE_POS_X as usize * FORWARD_CHANNELS as usize + local_channel;
    let active_channel = channel < channels;
    let active = active_channel && n < state_dim;
    let state_base = (batch * channels + channel) * state_dim;
    let a_base = channel * state_dim;
    let checkpoint_count = (seq_len + checkpoint_interval - 1) / checkpoint_interval;
    let mut state = if active { h_in[state_base + n] } else { 0.0f32 };

    for t in 0..seq_len {
        let btc = (batch * seq_len + t) * channels + channel;
        let btn = (batch * seq_len + t) * state_dim;
        let dt = if active_channel { delta[btc] } else { 0.0f32 };
        let x = if active_channel { xs[btc] } else { 0.0f32 };
        let contribution = if active {
            state = state * (dt * a[a_base + n]).exp() + dt * b_mat[btn + n] * x;
            state * c_mat[btn + n]
        } else {
            0.0f32
        };
        let out = plane_sum(contribution);
        if n == 0 && active_channel {
            y[btc] = out + x * d[channel];
        }

        if active && save_checkpoints && ((t + 1) % checkpoint_interval == 0 || t + 1 == seq_len) {
            let checkpoint = t / checkpoint_interval;
            let checkpoint_base =
                ((batch * checkpoint_count + checkpoint) * channels + channel) * state_dim;
            checkpoints[checkpoint_base + n] = state;
        }
    }
    if active {
        h_out[state_base + n] = state;
    }
}

/// Checkpointed forward: one block owns a `(batch, channel tile)` and
/// sweeps the segments left-to-right with the running state carried in
/// registers — a single launch and a single read of every input element,
/// replacing the partials/carry/apply chain that scanned the sequence
/// twice. The per-`(channel, state)` thread layout keeps the serial
/// recurrence with 16-wide state parallelism that A100 measurements
/// favor; the cross-state reduction for `y` lands in disjoint per-warp
/// slots flushed every `BACKWARD_FLUSH` steps, exactly like the reverse
/// sweep's gradient flush.
#[allow(clippy::manual_div_ceil, clippy::manual_is_multiple_of)]
#[cube(launch, fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
fn selective_scan_forward_swept<F: Float>(
    delta_raw: &Tensor<F>,
    xs: &Tensor<F>,
    b_mat: &Tensor<F>,
    c_mat: &Tensor<F>,
    a: &Tensor<f32>,
    d: &Tensor<f32>,
    h_in: &Tensor<f32>,
    y: &mut Tensor<F>,
    checkpoints: &mut Tensor<f32>,
    h_out: &mut Tensor<f32>,
    channels: u32,
    seq_len: u32,
    #[comptime] state_dim: usize,
    #[comptime] segment_len: usize,
) {
    let channels = channels as usize;
    let seq_len = seq_len as usize;
    let local_channel = UNIT_POS_X as usize;
    let n = UNIT_POS_Y as usize;
    let batch = CUBE_POS_Y as usize;
    let channel = CUBE_POS_X as usize * BACKWARD_CHANNELS + local_channel;
    let active_channel = channel < channels;
    let active = active_channel && n < state_dim;
    let tid = n * BACKWARD_CHANNELS + local_channel;
    let block_threads = BACKWARD_CHANNELS * state_dim;
    let state_index = (batch * channels + channel) * state_dim + n;
    let a_index = channel * state_dim + n;
    let segments = (seq_len + segment_len - 1) / segment_len;

    let tile_len = segment_len * BACKWARD_CHANNELS;
    let mut delta_tile = Shared::new_slice(tile_len);
    let mut xs_tile = Shared::new_slice(tile_len);
    let state_tile_len = segment_len * state_dim;
    let mut b_tile = Shared::new_slice(state_tile_len);
    let mut c_tile = Shared::new_slice(state_tile_len);
    let warp_rows = (state_dim + 1) / 2;
    let mut y_part = Shared::new_slice(BACKWARD_FLUSH * warp_rows * BACKWARD_CHANNELS);

    let av = if active { a[a_index] } else { 0.0f32 };
    let mut state = if active { h_in[state_index] } else { 0.0f32 };

    for segment in 0..segments {
        let start = segment * segment_len;
        let mut end = start + segment_len;
        if end > seq_len {
            end = seq_len;
        }
        let chunk_len = end - start;

        let mut index = tid;
        while index < tile_len {
            let t_local = index / BACKWARD_CHANNELS;
            let ch = CUBE_POS_X as usize * BACKWARD_CHANNELS + index % BACKWARD_CHANNELS;
            let mut raw = 0.0f32;
            let mut x = 0.0f32;
            if t_local < chunk_len && ch < channels {
                let btc = (batch * seq_len + start + t_local) * channels + ch;
                raw = f32::cast_from(delta_raw[btc]);
                x = f32::cast_from(xs[btc]);
            }
            delta_tile[index] = stable_softplus(raw);
            xs_tile[index] = x;
            index += block_threads;
        }
        let mut index = tid;
        while index < state_tile_len {
            let t_local = index / state_dim;
            let st = index % state_dim;
            let mut bv = 0.0f32;
            let mut cv = 0.0f32;
            if t_local < chunk_len {
                let btn = (batch * seq_len + start + t_local) * state_dim + st;
                bv = f32::cast_from(b_mat[btn]);
                cv = f32::cast_from(c_mat[btn]);
            }
            b_tile[index] = bv;
            c_tile[index] = cv;
            index += block_threads;
        }
        sync_cube();

        let flush_windows = segment_len / BACKWARD_FLUSH;
        #[unroll]
        for window in 0..flush_windows {
            #[unroll]
            for step in 0..BACKWARD_FLUSH {
                let offset = window * BACKWARD_FLUSH + step;
                let slot = offset % BACKWARD_FLUSH;
                if offset < chunk_len {
                    let cidx = offset * BACKWARD_CHANNELS + local_channel;
                    let nidx = offset * state_dim + n;
                    let dt = delta_tile[cidx];
                    let x = xs_tile[cidx];
                    let bv = b_tile[nidx];
                    let cv = c_tile[nidx];
                    state = state * (dt * av).exp() + dt * bv * x;
                    let y_term = state * cv;
                    // A warp spans two state rows of the channel tile;
                    // fold the odd row into the even one so each warp
                    // writes one disjoint partial row per step.
                    let mut partner = plane_shuffle_down(y_term, BACKWARD_CHANNELS as u32);
                    if n + 1 >= state_dim {
                        partner = 0.0f32;
                    }
                    if n % 2 == 0 {
                        let part = (slot * warp_rows + n / 2) * BACKWARD_CHANNELS + local_channel;
                        y_part[part] = y_term + partner;
                    }
                }
            }
            sync_cube();

            let window_lo = window * BACKWARD_FLUSH;
            let mut index = tid;
            while index < BACKWARD_FLUSH * BACKWARD_CHANNELS {
                let slot = index / BACKWARD_CHANNELS;
                let c_local = index % BACKWARD_CHANNELS;
                let offset = window_lo + slot;
                let ch = CUBE_POS_X as usize * BACKWARD_CHANNELS + c_local;
                if offset < chunk_len && ch < channels {
                    let mut y_sum = 0.0f32;
                    #[unroll]
                    for warp_row in 0..warp_rows {
                        let part = (slot * warp_rows + warp_row) * BACKWARD_CHANNELS + c_local;
                        y_sum += y_part[part];
                    }
                    let tidx = offset * BACKWARD_CHANNELS + c_local;
                    let btc = (batch * seq_len + start + offset) * channels + ch;
                    y[btc] = F::cast_from(y_sum + xs_tile[tidx] * d[ch]);
                }
                index += block_threads;
            }
            sync_cube();
        }

        if active {
            checkpoints[((batch * segments + segment) * channels + channel) * state_dim + n] =
                state;
        }
    }

    if active {
        h_out[state_index] = state;
    }
}

/// Checkpointed backward: one block owns a `(batch, channel tile)` and
/// sweeps the segments right-to-left with the adjoint carried in
/// registers — the exact reverse recurrence, no stitched-carry
/// approximation and no partials/carry launches. Each segment re-derives
/// its forward states from its checkpoint. The channel-grouped block
/// shape is deliberate: it buys the 16:1 cross-channel pre-reduction of
/// `grad_B`/`grad_C` before the global atomics, which a sequence-major
/// thread layout would forfeit.
#[allow(clippy::manual_div_ceil, clippy::useless_conversion)]
// `n % 2` must stay literal: the cube macro has no expansion for
// `is_multiple_of`.
#[allow(clippy::manual_is_multiple_of)]
#[cube(launch, fast_math = FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf)]
fn selective_scan_backward_segmented<F: Float>(
    delta_raw: &Tensor<F>,
    xs: &Tensor<F>,
    b_mat: &Tensor<F>,
    c_mat: &Tensor<F>,
    a: &Tensor<f32>,
    d: &Tensor<f32>,
    h_in: &Tensor<f32>,
    checkpoints: &Tensor<f32>,
    grad_y: &Tensor<F>,
    grad_delta: &mut Tensor<F>,
    grad_xs: &mut Tensor<F>,
    grad_b: &mut Tensor<Atomic<f32>>,
    grad_c: &mut Tensor<Atomic<f32>>,
    grad_a: &mut Tensor<Atomic<f32>>,
    grad_d: &mut Tensor<Atomic<f32>>,
    grad_h: &mut Tensor<f32>,
    channels: u32,
    seq_len: u32,
    #[comptime] state_dim: usize,
    #[comptime] segment_len: usize,
) {
    let channels = channels as usize;
    let seq_len = seq_len as usize;
    let local_channel = UNIT_POS_X as usize;
    let n = UNIT_POS_Y as usize;
    let batch = CUBE_POS_Y as usize;
    let channel = CUBE_POS_X as usize * BACKWARD_CHANNELS + local_channel;
    let active_channel = channel < channels;
    let active = active_channel && n < state_dim;
    let tid = n * BACKWARD_CHANNELS + local_channel;
    let block_threads = BACKWARD_CHANNELS * state_dim;
    let state_index = (batch * channels + channel) * state_dim + n;
    let a_index = channel * state_dim + n;
    let segments = (seq_len + segment_len - 1) / segment_len;

    // Every segment's sequence tiles stage through workgroup memory once;
    // both the rebuild and the reverse sweep read them from there, and
    // softplus runs in-kernel so the launcher never materializes a full
    // [batch, seq, channels] activation. Dead slots (sequence tail,
    // channel-tile overhang) load zeros; every consumer term multiplies
    // by an adjoint or gradient that is exactly zero there.
    let tile_len = segment_len * BACKWARD_CHANNELS;
    let mut raw_tile = Shared::new_slice(tile_len);
    let mut delta_tile = Shared::new_slice(tile_len);
    let mut xs_tile = Shared::new_slice(tile_len);
    let mut dy_tile = Shared::new_slice(tile_len);
    let state_tile_len = segment_len * state_dim;
    let mut b_tile = Shared::new_slice(state_tile_len);
    let mut c_tile = Shared::new_slice(state_tile_len);
    // Cross-thread reductions land in disjoint per-warp slots covering a
    // window of `BACKWARD_FLUSH` steps: two barriers per window instead
    // of one per step, and small enough for Metal's 32KB threadgroups.
    let warp_rows = (state_dim + 1) / 2;
    let mut dt_part = Shared::new_slice(BACKWARD_FLUSH * warp_rows * BACKWARD_CHANNELS);
    let mut dx_part = Shared::new_slice(BACKWARD_FLUSH * warp_rows * BACKWARD_CHANNELS);
    let mut gb_acc = Shared::new_slice(BACKWARD_FLUSH * state_dim);
    let mut gc_acc = Shared::new_slice(BACKWARD_FLUSH * state_dim);

    let av = if active { a[a_index] } else { 0.0f32 };
    // The adjoint is exactly zero at the right sequence boundary and
    // rides in a register across the whole reverse sweep.
    let mut adjoint = 0.0f32;
    let mut grad_a_local = 0.0f32;
    let mut grad_d_local = 0.0f32;

    for segment_rev in 0..segments {
        let segment = segments - 1 - segment_rev;
        let start = segment * segment_len;
        let mut end = start + segment_len;
        if end > seq_len {
            end = seq_len;
        }
        let chunk_len = end - start;

        let mut index = tid;
        while index < tile_len {
            let t_local = index / BACKWARD_CHANNELS;
            let ch = CUBE_POS_X as usize * BACKWARD_CHANNELS + index % BACKWARD_CHANNELS;
            let mut raw = 0.0f32;
            let mut x = 0.0f32;
            let mut dy = 0.0f32;
            if t_local < chunk_len && ch < channels {
                let btc = (batch * seq_len + start + t_local) * channels + ch;
                raw = f32::cast_from(delta_raw[btc]);
                x = f32::cast_from(xs[btc]);
                dy = f32::cast_from(grad_y[btc]);
            }
            raw_tile[index] = raw;
            delta_tile[index] = stable_softplus(raw);
            xs_tile[index] = x;
            dy_tile[index] = dy;
            index += block_threads;
        }
        let mut index = tid;
        while index < state_tile_len {
            let t_local = index / state_dim;
            let state = index % state_dim;
            let mut bv = 0.0f32;
            let mut cv = 0.0f32;
            if t_local < chunk_len {
                let btn = (batch * seq_len + start + t_local) * state_dim + state;
                bv = f32::cast_from(b_mat[btn]);
                cv = f32::cast_from(c_mat[btn]);
            }
            b_tile[index] = bv;
            c_tile[index] = cv;
            index += block_threads;
        }
        sync_cube();

        // Rebuild the segment's states from the entering checkpoint. The
        // fully unrolled loop keeps `chunk_states` register-resident
        // instead of spilling a per-thread array to local memory.
        // Inactive channels hold zeros: their `a` row loads as zero, so
        // the recurrence is a fixed point at zero and never overflows.
        let state_before = if active {
            if segment == 0 {
                h_in[state_index]
            } else {
                checkpoints[((batch * segments + segment - 1) * channels + channel) * state_dim + n]
            }
        } else {
            0.0f32
        };
        let mut state = state_before;
        let mut chunk_states = Array::<f32>::new(segment_len);
        #[unroll]
        for offset in 0..segment_len {
            if offset < chunk_len {
                let cidx = offset * BACKWARD_CHANNELS + local_channel;
                let dt = delta_tile[cidx];
                let x = xs_tile[cidx];
                let bv = b_tile[offset * state_dim + n];
                state = state * (dt * av).exp() + dt * bv * x;
                chunk_states[offset] = state;
            }
        }

        let flush_windows = segment_len / BACKWARD_FLUSH;
        #[unroll]
        for window in 0..flush_windows {
            #[unroll]
            for step in 0..BACKWARD_FLUSH {
                let offset = segment_len - 1 - (window * BACKWARD_FLUSH + step);
                // Windows are BACKWARD_FLUSH-aligned, so the accumulator
                // slot is just the offset's position within its window.
                let slot = offset % BACKWARD_FLUSH;
                if offset < chunk_len {
                    let cidx = offset * BACKWARD_CHANNELS + local_channel;
                    let nidx = offset * state_dim + n;
                    let dt = delta_tile[cidx];
                    let x = xs_tile[cidx];
                    let dy = dy_tile[cidx];
                    let bv = b_tile[nidx];
                    let cv = c_tile[nidx];
                    let alpha = (dt * av).exp();
                    let mut h_prev = state_before;
                    if offset > 0 {
                        h_prev = chunk_states[offset - 1];
                    }
                    let h_t = chunk_states[offset];
                    let g = adjoint + dy * cv;
                    let dt_term = g * (h_prev * alpha * av + bv * x);
                    let dx_term = g * dt * bv;
                    // A warp spans two state rows of the channel tile;
                    // fold the odd row into the even one so each warp
                    // writes one disjoint partial row per step — no
                    // barrier needed.
                    let mut partner_dt = plane_shuffle_down(dt_term, BACKWARD_CHANNELS as u32);
                    let mut partner_dx = plane_shuffle_down(dx_term, BACKWARD_CHANNELS as u32);
                    if n + 1 >= state_dim {
                        partner_dt = 0.0f32;
                        partner_dx = 0.0f32;
                    }
                    if n % 2 == 0 {
                        let part = (slot * warp_rows + n / 2) * BACKWARD_CHANNELS + local_channel;
                        dt_part[part] = dt_term + partner_dt;
                        dx_part[part] = dx_term + partner_dx;
                    }
                    let grad_b_sum = half_plane_sum(g * dt * x, local_channel);
                    let grad_c_sum = half_plane_sum(dy * h_t, local_channel);
                    if local_channel == 0 {
                        gb_acc[slot * state_dim + n] = grad_b_sum;
                        gc_acc[slot * state_dim + n] = grad_c_sum;
                    }
                    grad_a_local += g * h_prev * alpha * dt;
                    if n == 0 && active_channel {
                        grad_d_local += dy * x;
                    }
                    adjoint = g * alpha;
                }
            }
            sync_cube();

            let window_lo = segment_len - (window + 1) * BACKWARD_FLUSH;
            let mut index = tid;
            while index < BACKWARD_FLUSH * BACKWARD_CHANNELS {
                let slot = index / BACKWARD_CHANNELS;
                let c_local = index % BACKWARD_CHANNELS;
                let offset = window_lo + slot;
                let ch = CUBE_POS_X as usize * BACKWARD_CHANNELS + c_local;
                if offset < chunk_len && ch < channels {
                    let mut dt_sum = 0.0f32;
                    let mut dx_sum = 0.0f32;
                    #[unroll]
                    for warp_row in 0..warp_rows {
                        let part = (slot * warp_rows + warp_row) * BACKWARD_CHANNELS + c_local;
                        dt_sum += dt_part[part];
                        dx_sum += dx_part[part];
                    }
                    let tidx = offset * BACKWARD_CHANNELS + c_local;
                    let btc = (batch * seq_len + start + offset) * channels + ch;
                    grad_delta[btc] = F::cast_from(dt_sum * stable_sigmoid(raw_tile[tidx]));
                    grad_xs[btc] = F::cast_from(dy_tile[tidx] * d[ch] + dx_sum);
                }
                index += block_threads;
            }
            let mut index = tid;
            while index < BACKWARD_FLUSH * state_dim {
                let slot = index / state_dim;
                let state = index % state_dim;
                let offset = window_lo + slot;
                if offset < chunk_len {
                    let btn = (batch * seq_len + start + offset) * state_dim + state;
                    atomic_add_f32(&mut grad_b[btn], gb_acc[slot * state_dim + state]);
                    atomic_add_f32(&mut grad_c[btn], gc_acc[slot * state_dim + state]);
                }
                index += block_threads;
            }
            sync_cube();
        }
    }

    if active {
        atomic_add_f32(&mut grad_a[a_index], grad_a_local);
        grad_h[state_index] = adjoint;
    }
    if n == 0 && active_channel {
        atomic_add_f32(&mut grad_d[channel], grad_d_local);
    }
}

/// Materializes the softplus activation for the non-segmented scan
/// inference paths (decode, prefill without states). The training
/// kernels compute softplus in-kernel from the raw delta and never
/// allocate this tensor.
fn materialize_softplus<R: CubeRuntime>(
    delta_raw: &CubeTensor<R>,
    batch: usize,
    seq_len: usize,
    channels: usize,
) -> CubeTensor<R> {
    let delta = empty_like_dtype(
        delta_raw,
        Shape::new([batch, seq_len, channels]),
        DType::F32,
    );
    let delta_total = (batch * seq_len * channels) as u32;
    softplus_forward::launch::<f32, R>(
        &delta_raw.client,
        CubeCount::Static(delta_total.div_ceil(THREADS_PER_CUBE), 1, 1),
        CubeDim::new_1d(THREADS_PER_CUBE),
        delta_raw.clone().into_tensor_arg(),
        delta.clone().into_tensor_arg(),
    );
    delta
}

impl<R: CubeRuntime> MambaBackend for CubeBackend<R> {
    fn selective_scan_inner(
        delta: CubeTensor<R>,
        xs: CubeTensor<R>,
        b_mat: CubeTensor<R>,
        c_mat: CubeTensor<R>,
        a: CubeTensor<R>,
        d: CubeTensor<R>,
        h: CubeTensor<R>,
        state_dim: usize,
        save_states: bool,
    ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
        let [batch, seq_len, channels] = xs.shape().dims();
        assert!(
            state_dim.is_power_of_two() && (4..=16).contains(&state_dim),
            "GPU selective scan requires a power-of-two state_dim in 4..=16, got \
             {state_dim}: the CUDA runtime displaces writes into state tensors with \
             other minor strides, and 32 exceeds Metal's threadgroup budget (see \
             docs/kernel-tuning-surface.md); the CPU reference backend supports any \
             state_dim"
        );
        let delta_raw = into_contiguous(delta);
        let xs = into_contiguous(xs);
        let b_mat = into_contiguous(b_mat);
        let c_mat = into_contiguous(c_mat);
        let a = into_contiguous(a);
        let d = into_contiguous(d);
        let h = into_contiguous(h);
        let io_dtype = xs.dtype;
        assert!(
            io_dtype == DType::F32 || io_dtype == DType::BF16,
            "selective scan supports F32 or BF16 sequence tensors, got {io_dtype:?}"
        );
        assert_eq!(delta_raw.dtype, io_dtype);
        assert_eq!(b_mat.dtype, io_dtype);
        assert_eq!(c_mat.dtype, io_dtype);
        let checkpoint_interval = CHECKPOINTED_SCAN_INTERVAL;
        let full_sequence_scan = save_states || seq_len > 1;
        // Training (states saved for backward) always takes the swept
        // checkpointed kernels — any batch, any sequence length; the
        // remaining paths serve inference (decode, prefill).
        let segmented = save_states;
        // Only the training sweep kernels are BF16-native. The
        // inference paths (decode, prefill without saved states)
        // normalize a BF16 stream to FP32 here and hand BF16 back at
        // the return.
        let normalize_fp32 = io_dtype == DType::BF16 && !segmented;
        let (delta_raw, xs, b_mat, c_mat, io_dtype) = if normalize_fp32 {
            (
                burn_cubecl::kernel::cast(delta_raw, DType::F32),
                burn_cubecl::kernel::cast(xs, DType::F32),
                burn_cubecl::kernel::cast(b_mat, DType::F32),
                burn_cubecl::kernel::cast(c_mat, DType::F32),
                DType::F32,
            )
        } else {
            (delta_raw, xs, b_mat, c_mat, io_dtype)
        };
        let y = empty_like(&xs, Shape::new([batch, seq_len, channels]));
        let h_out = empty_like_dtype(&xs, Shape::new([batch, channels, state_dim]), DType::F32);
        let states = if save_states {
            empty_like_dtype(
                &xs,
                Shape::new([
                    batch,
                    seq_len.div_ceil(checkpoint_interval),
                    channels,
                    state_dim,
                ]),
                DType::F32,
            )
        } else {
            h_out.clone()
        };

        let client = xs.client.clone();
        if segmented {
            // Training path: one block per (batch, channel tile) sweeps
            // the segments left-to-right with the state in registers —
            // a single launch and a single input read, no stitched-carry
            // kernels.
            macro_rules! launch_forward {
                ($float:ty) => {{
                    selective_scan_forward_swept::launch::<$float, R>(
                        &client,
                        CubeCount::Static(
                            (channels as u32).div_ceil(BACKWARD_CHANNELS as u32),
                            batch as u32,
                            1,
                        ),
                        CubeDim::new_2d(BACKWARD_CHANNELS as u32, state_dim as u32),
                        delta_raw.clone().into_tensor_arg(),
                        xs.clone().into_tensor_arg(),
                        b_mat.clone().into_tensor_arg(),
                        c_mat.into_tensor_arg(),
                        a.clone().into_tensor_arg(),
                        d.into_tensor_arg(),
                        h.clone().into_tensor_arg(),
                        y.clone().into_tensor_arg(),
                        states.clone().into_tensor_arg(),
                        h_out.clone().into_tensor_arg(),
                        channels as u32,
                        seq_len as u32,
                        state_dim,
                        checkpoint_interval,
                    );
                }};
            }
            match io_dtype {
                DType::BF16 => launch_forward!(bf16),
                _ => launch_forward!(f32),
            }
        } else if full_sequence_scan {
            assert!(
                io_dtype == DType::F32,
                "BF16 selective scan requires the checkpointed training path"
            );
            let delta = materialize_softplus(&delta_raw, batch, seq_len, channels);
            let serial_blocks = ((batch * channels) as u32).div_ceil(THREADS_PER_CUBE);
            if serial_blocks >= SERIAL_SCAN_MIN_BLOCKS {
                selective_scan_forward_serial::launch::<R>(
                    &client,
                    CubeCount::Static(serial_blocks, 1, 1),
                    CubeDim::new_1d(THREADS_PER_CUBE),
                    delta.clone().into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.clone().into_tensor_arg(),
                    c_mat.into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    d.into_tensor_arg(),
                    h.clone().into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    states.clone().into_tensor_arg(),
                    h_out.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                    save_states,
                );
            } else {
                selective_scan_forward_parallel::launch::<R>(
                    &client,
                    CubeCount::Static(
                        (channels as u32).div_ceil(FORWARD_CHANNELS),
                        batch as u32,
                        1,
                    ),
                    CubeDim::new_2d(PLANE_WIDTH, FORWARD_CHANNELS),
                    delta.clone().into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.clone().into_tensor_arg(),
                    c_mat.into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    d.into_tensor_arg(),
                    h.clone().into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    states.clone().into_tensor_arg(),
                    h_out.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    checkpoint_interval,
                    save_states,
                );
            }
        } else {
            assert!(
                io_dtype == DType::F32,
                "BF16 selective scan requires the checkpointed training path"
            );
            let delta = materialize_softplus(&delta_raw, batch, seq_len, channels);
            let total = (batch * channels) as u32;
            selective_scan_step::launch::<R>(
                &client,
                CubeCount::Static(total.div_ceil(THREADS_PER_CUBE), 1, 1),
                CubeDim::new_1d(THREADS_PER_CUBE),
                delta.into_tensor_arg(),
                xs.clone().into_tensor_arg(),
                b_mat.into_tensor_arg(),
                c_mat.into_tensor_arg(),
                a.into_tensor_arg(),
                d.into_tensor_arg(),
                h.into_tensor_arg(),
                y.clone().into_tensor_arg(),
                h_out.clone().into_tensor_arg(),
                channels as u32,
                state_dim,
            );
        }

        let y = if normalize_fp32 {
            burn_cubecl::kernel::cast(y, DType::BF16)
        } else {
            y
        };
        (y, h_out, states)
    }

    fn selective_scan_backward(
        delta: CubeTensor<R>,
        xs: CubeTensor<R>,
        b_mat: CubeTensor<R>,
        c_mat: CubeTensor<R>,
        a: CubeTensor<R>,
        d: CubeTensor<R>,
        h: CubeTensor<R>,
        states: CubeTensor<R>,
        grad_y: CubeTensor<R>,
        state_dim: usize,
    ) -> (
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    ) {
        let [batch, seq_len, channels] = xs.shape().dims();
        assert!(
            state_dim.is_power_of_two() && (4..=16).contains(&state_dim),
            "GPU selective scan requires a power-of-two state_dim in 4..=16, got \
             {state_dim}: the CUDA runtime displaces writes into state tensors with \
             other minor strides, and 32 exceeds Metal's threadgroup budget (see \
             docs/kernel-tuning-surface.md); the CPU reference backend supports any \
             state_dim"
        );
        let delta_raw = into_contiguous(delta);
        let xs = into_contiguous(xs);
        let b_mat = into_contiguous(b_mat);
        let c_mat = into_contiguous(c_mat);
        let a = into_contiguous(a);
        let d = into_contiguous(d);
        let h = into_contiguous(h);
        let states = into_contiguous(states);
        let grad_y = into_contiguous(grad_y);
        let io_dtype = xs.dtype;
        assert!(
            io_dtype == DType::F32 || io_dtype == DType::BF16,
            "selective scan supports F32 or BF16 sequence tensors, got {io_dtype:?}"
        );
        assert_eq!(
            grad_y.dtype, io_dtype,
            "selective scan output gradient dtype must match the sequence dtype"
        );
        let grad_delta = empty_like(&xs, Shape::new([batch, seq_len, channels]));
        let grad_xs = empty_like(&xs, Shape::new([batch, seq_len, channels]));
        let grad_b = zeros_like_dtype(&xs, Shape::new([batch, seq_len, state_dim]), DType::F32);
        let grad_c = zeros_like_dtype(&xs, Shape::new([batch, seq_len, state_dim]), DType::F32);
        let grad_a = zeros_like_dtype(&xs, Shape::new([channels, state_dim]), DType::F32);
        let grad_d = zeros_like_dtype(&xs, Shape::new([channels]), DType::F32);
        let grad_h = empty_like_dtype(&xs, Shape::new([batch, channels, state_dim]), DType::F32);
        let client = xs.client.clone();

        // One block per (batch, channel tile) sweeps the checkpoint
        // segments right-to-left with the adjoint in registers — the
        // exact reverse recurrence, native in both F32 and BF16.
        macro_rules! launch_backward {
            ($float:ty) => {{
                selective_scan_backward_segmented::launch::<$float, R>(
                    &client,
                    CubeCount::Static(
                        (channels as u32).div_ceil(BACKWARD_CHANNELS as u32),
                        batch as u32,
                        1,
                    ),
                    CubeDim::new_2d(BACKWARD_CHANNELS as u32, state_dim as u32),
                    delta_raw.into_tensor_arg(),
                    xs.clone().into_tensor_arg(),
                    b_mat.into_tensor_arg(),
                    c_mat.clone().into_tensor_arg(),
                    a.clone().into_tensor_arg(),
                    d.into_tensor_arg(),
                    h.clone().into_tensor_arg(),
                    states.clone().into_tensor_arg(),
                    grad_y.clone().into_tensor_arg(),
                    grad_delta.clone().into_tensor_arg(),
                    grad_xs.clone().into_tensor_arg(),
                    grad_b.clone().into_tensor_arg(),
                    grad_c.clone().into_tensor_arg(),
                    grad_a.clone().into_tensor_arg(),
                    grad_d.clone().into_tensor_arg(),
                    grad_h.clone().into_tensor_arg(),
                    channels as u32,
                    seq_len as u32,
                    state_dim,
                    CHECKPOINTED_SCAN_INTERVAL,
                );
            }};
        }
        match io_dtype {
            DType::BF16 => launch_backward!(bf16),
            _ => launch_backward!(f32),
        }
        // grad_B/grad_C accumulate through f32 atomics; hand them back
        // in the sequence dtype so autodiff composes without dtype
        // mismatches. The tensors are [batch, seq, state_dim] — tiny.
        let (grad_b, grad_c) = if io_dtype != DType::F32 {
            (
                burn_cubecl::kernel::cast(grad_b, io_dtype),
                burn_cubecl::kernel::cast(grad_c, io_dtype),
            )
        } else {
            (grad_b, grad_c)
        };
        (grad_delta, grad_xs, grad_b, grad_c, grad_a, grad_d, grad_h)
    }
}
