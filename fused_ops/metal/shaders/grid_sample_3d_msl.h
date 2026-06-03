// Copyright (c) 2026 Rohit Jena. All rights reserved.
//
// This file is part of FireANTs, distributed under the terms of
// the FireANTs License version 1.0. A copy of the license can be found
// in the LICENSE file at the root of this repository.
//
// IMPORTANT: This code is part of FireANTs and its use, reproduction, or
// distribution must comply with the full license terms, including:
// - Maintaining all copyright notices and bibliography references
// - Using only approved (re)-distribution channels
// - Proper attribution in derivative works
//
// For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE

#pragma once

// Backward kernel for torch.nn.functional.grid_sample on 5D input
// (mode='bilinear', padding_mode='zeros'). One thread per output position
// (n, d_out, h_out, w_out); loops over channels internally.
//
// Atomic-add into grad_input uses a CAS loop on a uint reinterpretation of
// the float buffer because Metal lacks native device-memory atomic_float add.
// grad_grid writes are conflict-free under this layout (one thread per grid
// position), so they are plain stores.
//
// Matches PyTorch's aten::grid_sampler_3d_backward semantics:
//   ix = ((x + 1) * (W - 1)) / 2          if align_corners
//        ((x + 1) * W - 1) / 2            otherwise
// and analogously for iy/iz. Out-of-bounds samples contribute zero (zeros
// padding) on both forward and backward.

namespace fireants_fused_ops::metal {

static const char* grid_sample_3d_backward_msl_src = R"METAL(
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Float atomic-add via CAS loop on uint reinterpretation.
inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + val;
        uint desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, desired_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
        // On failure, `expected` was overwritten with the latest value; retry.
    }
}

struct GridSample3dParams {
    int N;
    int C;
    int D_in;
    int H_in;
    int W_in;
    int D_out;
    int H_out;
    int W_out;
    int align_corners;   // 0 or 1
};

inline float coord_to_index(float coord, int size, bool align_corners) {
    if (align_corners) {
        return ((coord + 1.0f) * 0.5f) * float(size - 1);
    } else {
        return ((coord + 1.0f) * float(size) - 1.0f) * 0.5f;
    }
}

inline float coord_to_index_deriv(int size, bool align_corners) {
    // d(index)/d(coord)
    return align_corners ? float(size - 1) * 0.5f : float(size) * 0.5f;
}

kernel void grid_sample_3d_backward(
    device const float*       grad_output  [[ buffer(0) ]],   // [N, C, D_out, H_out, W_out]
    device const float*       input        [[ buffer(1) ]],   // [N, C, D_in, H_in, W_in]
    device const float*       grid         [[ buffer(2) ]],   // [N, D_out, H_out, W_out, 3]
    device atomic_uint*       grad_input   [[ buffer(3) ]],   // [N, C, D_in, H_in, W_in], aliased
    device float*             grad_grid    [[ buffer(4) ]],   // [N, D_out, H_out, W_out, 3]
    constant GridSample3dParams& P         [[ buffer(5) ]],
    uint                      tid          [[ thread_position_in_grid ]],
    uint                      gsize        [[ threads_per_grid ]])
{
    const int total = P.N * P.D_out * P.H_out * P.W_out;
    const bool align_corners = (P.align_corners != 0);
    const int CHW_in  = P.C * P.D_in  * P.H_in  * P.W_in;
    const int DHW_in  = P.D_in * P.H_in * P.W_in;
    const int HW_in   = P.H_in * P.W_in;
    const int CHW_out = P.C * P.D_out * P.H_out * P.W_out;
    const int DHW_out = P.D_out * P.H_out * P.W_out;
    const int HW_out  = P.H_out * P.W_out;

    for (int idx = int(tid); idx < total; idx += int(gsize)) {
        // Decompose idx -> (n, d_out, h_out, w_out)
        int w_out = idx % P.W_out;
        int rem1  = idx / P.W_out;
        int h_out = rem1 % P.H_out;
        int rem2  = rem1 / P.H_out;
        int d_out = rem2 % P.D_out;
        int n     = rem2 / P.D_out;

        const int grid_off = ((n * P.D_out + d_out) * P.H_out + h_out) * P.W_out + w_out;
        float x = grid[grid_off * 3 + 0];
        float y = grid[grid_off * 3 + 1];
        float z = grid[grid_off * 3 + 2];

        float ix = coord_to_index(x, P.W_in, align_corners);
        float iy = coord_to_index(y, P.H_in, align_corners);
        float iz = coord_to_index(z, P.D_in, align_corners);

        int ix0 = int(floor(ix));
        int iy0 = int(floor(iy));
        int iz0 = int(floor(iz));
        int ix1 = ix0 + 1;
        int iy1 = iy0 + 1;
        int iz1 = iz0 + 1;

        // Trilinear weights (per axis fractional)
        float tx = ix - float(ix0);
        float ty = iy - float(iy0);
        float tz = iz - float(iz0);
        float ax0 = 1.0f - tx, ax1 = tx;
        float ay0 = 1.0f - ty, ay1 = ty;
        float az0 = 1.0f - tz, az1 = tz;

        // 8 corner weights, indexed [k] where bit0=x, bit1=y, bit2=z
        float w_corner[8] = {
            ax0 * ay0 * az0,  // 000
            ax1 * ay0 * az0,  // 100
            ax0 * ay1 * az0,  // 010
            ax1 * ay1 * az0,  // 110
            ax0 * ay0 * az1,  // 001
            ax1 * ay0 * az1,  // 101
            ax0 * ay1 * az1,  // 011
            ax1 * ay1 * az1,  // 111
        };
        int cx[8] = { ix0, ix1, ix0, ix1, ix0, ix1, ix0, ix1 };
        int cy[8] = { iy0, iy0, iy1, iy1, iy0, iy0, iy1, iy1 };
        int cz[8] = { iz0, iz0, iz0, iz0, iz1, iz1, iz1, iz1 };
        bool in[8];
        for (int k = 0; k < 8; ++k) {
            in[k] = (cx[k] >= 0 && cx[k] < P.W_in &&
                     cy[k] >= 0 && cy[k] < P.H_in &&
                     cz[k] >= 0 && cz[k] < P.D_in);
        }

        // d(weight)/d(ix), d/d(iy), d/d(iz) per corner.
        // d ax0/dix = -1, d ax1/dix = +1, etc.
        float dw_dx[8] = {
            (-1.0f) * ay0 * az0,
            (+1.0f) * ay0 * az0,
            (-1.0f) * ay1 * az0,
            (+1.0f) * ay1 * az0,
            (-1.0f) * ay0 * az1,
            (+1.0f) * ay0 * az1,
            (-1.0f) * ay1 * az1,
            (+1.0f) * ay1 * az1,
        };
        float dw_dy[8] = {
            ax0 * (-1.0f) * az0,
            ax1 * (-1.0f) * az0,
            ax0 * (+1.0f) * az0,
            ax1 * (+1.0f) * az0,
            ax0 * (-1.0f) * az1,
            ax1 * (-1.0f) * az1,
            ax0 * (+1.0f) * az1,
            ax1 * (+1.0f) * az1,
        };
        float dw_dz[8] = {
            ax0 * ay0 * (-1.0f),
            ax1 * ay0 * (-1.0f),
            ax0 * ay1 * (-1.0f),
            ax1 * ay1 * (-1.0f),
            ax0 * ay0 * (+1.0f),
            ax1 * ay0 * (+1.0f),
            ax0 * ay1 * (+1.0f),
            ax1 * ay1 * (+1.0f),
        };

        // Accumulate grad_grid contributions over channels.
        float ggx = 0.0f, ggy = 0.0f, ggz = 0.0f;

        for (int c = 0; c < P.C; ++c) {
            const int go_off = ((n * P.C + c) * P.D_out + d_out) * P.H_out * P.W_out
                                + h_out * P.W_out + w_out;
            float go = grad_output[go_off];

            for (int k = 0; k < 8; ++k) {
                if (!in[k]) continue;
                const int in_off = ((n * P.C + c) * P.D_in + cz[k]) * HW_in
                                    + cy[k] * P.W_in + cx[k];
                // grad_input += go * weight
                atomic_add_float(grad_input + in_off, go * w_corner[k]);
                // grad_grid contribution: go * input[in_off] * dw/dcoord
                float iv = input[in_off];
                ggx += go * iv * dw_dx[k];
                ggy += go * iv * dw_dy[k];
                ggz += go * iv * dw_dz[k];
            }
        }

        // chain rule: d(out)/d(grid) = d(out)/d(index) * d(index)/d(coord)
        ggx *= coord_to_index_deriv(P.W_in, align_corners);
        ggy *= coord_to_index_deriv(P.H_in, align_corners);
        ggz *= coord_to_index_deriv(P.D_in, align_corners);

        grad_grid[grid_off * 3 + 0] = ggx;
        grad_grid[grid_off * 3 + 1] = ggy;
        grad_grid[grid_off * 3 + 2] = ggz;
    }
}
)METAL";

} // namespace fireants_fused_ops::metal
