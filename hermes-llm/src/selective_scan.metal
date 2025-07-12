// Fused Mamba selective scan — one dispatch for the whole scan.
// Design: docs/metal-selective-scan.md. Buffer packing must match
// src/metal_scan.rs (FusedSelectiveScan).
//
// Grid: one thread per (batch, channel); each thread keeps its h[N] state in
// registers and walks t = 0..L sequentially (the scan's data dependence).
#include <metal_stdlib>
using namespace metal;

constant constexpr uint MAX_STATE_DIM = 64;

struct ScanParams {
    uint batch;
    uint seq_len;
    uint d_inner;
    uint state_dim;
};

kernel void fused_selective_scan_f32(
    constant ScanParams &p    [[buffer(0)]],
    const device float *xd    [[buffer(1)]], // [B, L, 2*di]: xs | delta
    const device float *bc    [[buffer(2)]], // [B, L, 2*N]: B | C
    const device float *adh   [[buffer(3)]], // A [di*N] | D [di] | h0 [B*di*N]
    device float *out         [[buffer(4)]], // y [B*L*di] | h_final [B*di*N]
    uint tid                  [[thread_position_in_grid]]
) {
    const uint total = p.batch * p.d_inner;
    if (tid >= total) {
        return;
    }
    const uint b = tid / p.d_inner;
    const uint c = tid % p.d_inner;
    const uint N = p.state_dim;

    const device float *A = adh + c * N;
    const float Dc = adh[p.d_inner * N + c];
    const device float *h0 = adh + p.d_inner * N + p.d_inner + (b * p.d_inner + c) * N;

    float h[MAX_STATE_DIM];
    for (uint n = 0; n < N; n++) {
        h[n] = h0[n];
    }

    const uint xd_row = 2 * p.d_inner;
    const uint bc_row = 2 * N;
    for (uint t = 0; t < p.seq_len; t++) {
        const device float *xrow = xd + (b * p.seq_len + t) * xd_row;
        const float x = xrow[c];
        const float dt = xrow[p.d_inner + c];
        const device float *brow = bc + (b * p.seq_len + t) * bc_row;
        float y = 0.0f;
        for (uint n = 0; n < N; n++) {
            const float da = exp(dt * A[n]);
            h[n] = h[n] * da + dt * brow[n] * x;
            y = fma(h[n], brow[N + n], y);
        }
        y = fma(Dc, x, y);
        out[(b * p.seq_len + t) * p.d_inner + c] = y;
    }

    device float *hout = out + p.batch * p.seq_len * p.d_inner + (b * p.d_inner + c) * N;
    for (uint n = 0; n < N; n++) {
        hout[n] = h[n];
    }
}

// Depthwise (groups == channels) causal conv1d, bias folded in.
//
// Candle's grouped conv1d splits into one conv per group — with
// groups == d_inner that is ~16k dispatches per decoded token across the
// Mamba layers (measured: 98% of decode CPU time). One dispatch instead.
struct ConvParams {
    uint batch;
    uint d_inner;
    uint l_in;
    uint ksize;
};

kernel void depthwise_conv1d_f32(
    constant ConvParams &p    [[buffer(0)]],
    const device float *in    [[buffer(1)]], // [B, di, L_in]
    const device float *w     [[buffer(2)]], // [di * K]
    const device float *bias  [[buffer(3)]], // [di]
    device float *out         [[buffer(4)]], // [B, di, L_in - K + 1]
    uint tid                  [[thread_position_in_grid]]
) {
    const uint total = p.batch * p.d_inner;
    if (tid >= total) {
        return;
    }
    const uint b = tid / p.d_inner;
    const uint c = tid % p.d_inner;
    const uint l_out = p.l_in - p.ksize + 1;

    const device float *row = in + (b * p.d_inner + c) * p.l_in;
    const device float *wc = w + c * p.ksize;
    device float *orow = out + (b * p.d_inner + c) * l_out;
    const float bc = bias[c];
    for (uint t = 0; t < l_out; t++) {
        float acc = bc;
        for (uint k = 0; k < p.ksize; k++) {
            acc = fma(wc[k], row[t + k], acc);
        }
        orow[t] = acc;
    }
}
