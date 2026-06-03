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

// Forward kernel for the FireANTs fused generic-label 3D grid sampler on MPS.
//
// Mirrors the CUDA implementation in
// fused_ops/src/FusedGridSamplerGenericLabel.cu (forward path) for float32
// inputs/grids. One thread per output spatial position (n, d_out, h_out, w_out);
// channels are looped inside the thread so the per-thread label-accumulation
// buffer can be reused across channels.
//
// At each output position the kernel:
//   1. Resolves a sampling coordinate (x, y, z) in input index space from
//      some combination of `affine_3d`, `grid`, and `grid_affine_pregrid`
//      depending on which buffers are non-null and `is_displacement`.
//   2. Applies the standard PyTorch grid_sampler coord transform (align_corners
//      + padding_mode: zeros/border/reflection).
//   3. Reads the 8 trilinear corners and trilinear weights.
//   4. For each input channel: accumulates per-unique-label weights across
//      the 8 corners (label equality with 1e-5 tolerance, up to 8 distinct
//      labels), then writes the argmax label (and optionally its weight).
//
// Buffer null-ness is encoded via a `flags` bitfield in the param block, since
// Metal cannot express optional pointers directly. Broadcast over the batch
// dimension for input / affine / grid / grid_affine is also encoded there.

namespace fireants_fused_ops::metal {

static const char* grid_sample_3d_generic_label_forward_msl_src = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Bit positions in GridSampleGL3dParams::flags
constant uint FLAG_HAS_AFFINE          = 1u << 0;
constant uint FLAG_HAS_GRID            = 1u << 1;
constant uint FLAG_HAS_GRID_AFFINE     = 1u << 2;
constant uint FLAG_IS_DISPLACEMENT     = 1u << 3;
constant uint FLAG_ALIGN_CORNERS       = 1u << 4;
constant uint FLAG_RETURN_WEIGHT       = 1u << 5;
constant uint FLAG_BROADCAST_INPUT     = 1u << 6;
constant uint FLAG_BROADCAST_AFFINE    = 1u << 7;
constant uint FLAG_BROADCAST_GRID      = 1u << 8;
constant uint FLAG_BROADCAST_GAFFINE   = 1u << 9;

// padding_mode integer values must match GRID_SAMPLE_PADDING_MODES in Python.
constant int PADDING_ZEROS      = 0;
constant int PADDING_BORDER     = 1;
constant int PADDING_REFLECTION = 2;

struct GridSampleGL3dParams {
    int N;
    int C;
    int D_in;
    int H_in;
    int W_in;
    int D_out;
    int H_out;
    int W_out;
    float grid_xmin;
    float grid_ymin;
    float grid_zmin;
    float grid_xmax;
    float grid_ymax;
    float grid_zmax;
    int   padding_mode;   // PADDING_*
    float dummy;          // background fill / "ignore" sentinel
    uint  flags;          // FLAG_* bitfield
};

inline float coord_to_index(float coord, int size, bool align_corners) {
    if (align_corners) {
        return ((coord + 1.0f) * 0.5f) * float(size - 1);
    } else {
        return ((coord + 1.0f) * float(size) - 1.0f) * 0.5f;
    }
}

inline float clip_coord(float in, int size) {
    return clamp(in, 0.0f, float(size - 1));
}

inline float reflect_coord(float in, float twice_low, float twice_high) {
    if (twice_low == twice_high) {
        return 0.0f;
    }
    float min_v = twice_low * 0.5f;
    float span = (twice_high - twice_low) * 0.5f;
    in = fabs(in - min_v);
    float extra = fmod(in, span);
    int flips = int(floor(in / span));
    if ((flips & 1) == 0) {
        return extra + min_v;
    } else {
        return span - extra + min_v;
    }
}

inline float apply_padding_mode(float idx, int size, int padding_mode, bool align_corners) {
    if (padding_mode == PADDING_BORDER) {
        return clip_coord(idx, size);
    }
    if (padding_mode == PADDING_REFLECTION) {
        if (align_corners) {
            idx = reflect_coord(idx, 0.0f, 2.0f * float(size - 1));
        } else {
            idx = reflect_coord(idx, -1.0f, 2.0f * float(size) - 1.0f);
        }
        // safety clamp in case of fp wobble at the edges
        return clip_coord(idx, size);
    }
    return idx;   // PADDING_ZEROS — let within-bounds checks filter samples
}

inline bool within_bounds_3d(int z, int y, int x, int Dz, int Hy, int Wx) {
    return (x >= 0) && (x < Wx) && (y >= 0) && (y < Hy) && (z >= 0) && (z < Dz);
}

kernel void grid_sample_3d_generic_label_forward(
    device const float*       input            [[ buffer(0) ]],
    device const float*       affine_3d        [[ buffer(1) ]],
    device const float*       grid             [[ buffer(2) ]],
    device const float*       grid_affine      [[ buffer(3) ]],
    device       float*       output_labels    [[ buffer(4) ]],
    device       float*       output_weights   [[ buffer(5) ]],
    constant GridSampleGL3dParams& P            [[ buffer(6) ]],
    uint                      tid              [[ thread_position_in_grid ]],
    uint                      gsize            [[ threads_per_grid ]])
{
    const int total = P.N * P.D_out * P.H_out * P.W_out;
    if (total <= 0) return;

    const bool has_affine        = (P.flags & FLAG_HAS_AFFINE)      != 0;
    const bool has_grid          = (P.flags & FLAG_HAS_GRID)        != 0;
    const bool has_grid_affine   = (P.flags & FLAG_HAS_GRID_AFFINE) != 0;
    const bool is_displacement   = (P.flags & FLAG_IS_DISPLACEMENT) != 0;
    const bool align_corners     = (P.flags & FLAG_ALIGN_CORNERS)   != 0;
    const bool return_weight     = (P.flags & FLAG_RETURN_WEIGHT)   != 0;
    const bool bcast_input       = (P.flags & FLAG_BROADCAST_INPUT) != 0;
    const bool bcast_affine      = (P.flags & FLAG_BROADCAST_AFFINE)  != 0;
    const bool bcast_grid        = (P.flags & FLAG_BROADCAST_GRID)    != 0;
    const bool bcast_grid_affine = (P.flags & FLAG_BROADCAST_GAFFINE) != 0;

    const int Wo = P.W_out;
    const int Ho = P.H_out;
    const int Do = P.D_out;
    const int Wi = P.W_in;
    const int Hi = P.H_in;
    const int Di = P.D_in;
    const int C  = P.C;

    const int inp_sC  = Di * Hi * Wi;          // stride between channels (input)
    const int out_sC  = Do * Ho * Wo;          // stride between channels (output)
    const int out_sN  = C * out_sC;            // stride between batches (output)
    const int inp_sN  = C * inp_sC;            // stride between batches (input)

    for (int idx = int(tid); idx < total; idx += int(gsize)) {
        const int w_out = idx % Wo;
        const int h_out = (idx / Wo) % Ho;
        const int d_out = (idx / (Wo * Ho)) % Do;
        const int n     = idx / (Wo * Ho * Do);

        // --- resolve sampling coordinate (in input grid normalised space) ---
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float ix_n, iy_n, iz_n;
        ix_n = float(w_out) * (P.grid_xmax - P.grid_xmin) / float(Wo - 1) + P.grid_xmin;
        iy_n = float(h_out) * (P.grid_ymax - P.grid_ymin) / float(Ho - 1) + P.grid_ymin;
        iz_n = float(d_out) * (P.grid_zmax - P.grid_zmin) / float(Do - 1) + P.grid_zmin;

        if (!has_grid) {
            // pure affine: x = A * [ix_n; iy_n; iz_n; 1]
            const int aff_n = bcast_affine ? 0 : (12 * n);
            x = affine_3d[aff_n + 0] * ix_n + affine_3d[aff_n + 1] * iy_n +
                affine_3d[aff_n + 2] * iz_n + affine_3d[aff_n + 3];
            y = affine_3d[aff_n + 4] * ix_n + affine_3d[aff_n + 5] * iy_n +
                affine_3d[aff_n + 6] * iz_n + affine_3d[aff_n + 7];
            z = affine_3d[aff_n + 8] * ix_n + affine_3d[aff_n + 9] * iy_n +
                affine_3d[aff_n + 10] * iz_n + affine_3d[aff_n + 11];
        } else {
            const int grid_n_offset = bcast_grid ? 0 : (Do * Ho * Wo * 3 * n);
            const int grid_off = grid_n_offset + 3 * (w_out + Wo * (h_out + Ho * d_out));
            float dx = grid[grid_off + 0];
            float dy = grid[grid_off + 1];
            float dz = grid[grid_off + 2];

            float i_dx, i_dy, i_dz;
            if (has_grid_affine) {
                const int pre_n = bcast_grid_affine ? 0 : (9 * n);
                i_dx = grid_affine[pre_n + 0] * dx + grid_affine[pre_n + 1] * dy + grid_affine[pre_n + 2] * dz;
                i_dy = grid_affine[pre_n + 3] * dx + grid_affine[pre_n + 4] * dy + grid_affine[pre_n + 5] * dz;
                i_dz = grid_affine[pre_n + 6] * dx + grid_affine[pre_n + 7] * dy + grid_affine[pre_n + 8] * dz;
            } else {
                i_dx = dx; i_dy = dy; i_dz = dz;
            }
            if (is_displacement) {
                if (has_affine) {
                    const int aff_n = bcast_affine ? 0 : (12 * n);
                    x = affine_3d[aff_n + 0] * ix_n + affine_3d[aff_n + 1] * iy_n +
                        affine_3d[aff_n + 2] * iz_n + affine_3d[aff_n + 3];
                    y = affine_3d[aff_n + 4] * ix_n + affine_3d[aff_n + 5] * iy_n +
                        affine_3d[aff_n + 6] * iz_n + affine_3d[aff_n + 7];
                    z = affine_3d[aff_n + 8] * ix_n + affine_3d[aff_n + 9] * iy_n +
                        affine_3d[aff_n + 10] * iz_n + affine_3d[aff_n + 11];
                } else {
                    x = ix_n; y = iy_n; z = iz_n;
                }
                x += i_dx; y += i_dy; z += i_dz;
            } else {
                x = i_dx; y = i_dy; z = i_dz;
            }
        }

        // --- normalised coords -> input index space + padding mode ---
        float ix = coord_to_index(x, Wi, align_corners);
        float iy = coord_to_index(y, Hi, align_corners);
        float iz = coord_to_index(z, Di, align_corners);
        ix = apply_padding_mode(ix, Wi, P.padding_mode, align_corners);
        iy = apply_padding_mode(iy, Hi, P.padding_mode, align_corners);
        iz = apply_padding_mode(iz, Di, P.padding_mode, align_corners);

        const int ix_tnw = int(floor(ix));
        const int iy_tnw = int(floor(iy));
        const int iz_tnw = int(floor(iz));
        const int ix_e = ix_tnw + 1, iy_e = iy_tnw + 1, iz_e = iz_tnw + 1;

        // trilinear weights — note the per-corner expression matches the CUDA kernel
        const float tnw = float(ix_e   - ix) * float(iy_e   - iy) * float(iz_e   - iz);
        const float tne = float(ix     - ix_tnw) * float(iy_e   - iy) * float(iz_e   - iz);
        const float tsw = float(ix_e   - ix) * float(iy     - iy_tnw) * float(iz_e   - iz);
        const float tse = float(ix     - ix_tnw) * float(iy     - iy_tnw) * float(iz_e   - iz);
        const float bnw = float(ix_e   - ix) * float(iy_e   - iy) * float(iz     - iz_tnw);
        const float bne = float(ix     - ix_tnw) * float(iy_e   - iy) * float(iz     - iz_tnw);
        const float bsw = float(ix_e   - ix) * float(iy     - iy_tnw) * float(iz     - iz_tnw);
        const float bse = float(ix     - ix_tnw) * float(iy     - iy_tnw) * float(iz     - iz_tnw);

        // --- per-channel argmax-over-unique-labels ---
        const int n_in = bcast_input ? 0 : n;
        const int batch_in_off = n_in * inp_sN;
        const int batch_out_off = n * out_sN;
        const float dummy = P.dummy;

        for (int c = 0; c < C; ++c) {
            const int ch_in_off = batch_in_off + c * inp_sC;

            // Read 8 corner values (or dummy-1 when out of bounds under zeros padding).
            // For BORDER / REFLECTION the apply_padding_mode step already clamped
            // ix/iy/iz so within_bounds_3d is true for all corners.
            float v0, v1, v2, v3, v4, v5, v6, v7;
            v0 = within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi) ? input[ch_in_off + ix_tnw + Wi * (iy_tnw + Hi * iz_tnw)] : (dummy - 1.0f);
            v1 = within_bounds_3d(iz_tnw, iy_tnw, ix_e,   Di, Hi, Wi) ? input[ch_in_off + ix_e   + Wi * (iy_tnw + Hi * iz_tnw)] : (dummy - 1.0f);
            v2 = within_bounds_3d(iz_tnw, iy_e,   ix_tnw, Di, Hi, Wi) ? input[ch_in_off + ix_tnw + Wi * (iy_e   + Hi * iz_tnw)] : (dummy - 1.0f);
            v3 = within_bounds_3d(iz_tnw, iy_e,   ix_e,   Di, Hi, Wi) ? input[ch_in_off + ix_e   + Wi * (iy_e   + Hi * iz_tnw)] : (dummy - 1.0f);
            v4 = within_bounds_3d(iz_e,   iy_tnw, ix_tnw, Di, Hi, Wi) ? input[ch_in_off + ix_tnw + Wi * (iy_tnw + Hi * iz_e  )] : (dummy - 1.0f);
            v5 = within_bounds_3d(iz_e,   iy_tnw, ix_e,   Di, Hi, Wi) ? input[ch_in_off + ix_e   + Wi * (iy_tnw + Hi * iz_e  )] : (dummy - 1.0f);
            v6 = within_bounds_3d(iz_e,   iy_e,   ix_tnw, Di, Hi, Wi) ? input[ch_in_off + ix_tnw + Wi * (iy_e   + Hi * iz_e  )] : (dummy - 1.0f);
            v7 = within_bounds_3d(iz_e,   iy_e,   ix_e,   Di, Hi, Wi) ? input[ch_in_off + ix_e   + Wi * (iy_e   + Hi * iz_e  )] : (dummy - 1.0f);

            // Accumulate weights per unique label (matches the CUDA add_or_accumulate
            // lambda — up to 8 distinct labels, equality tolerance 1e-5).
            float val_list[8];
            float weight_list[8];
            int nvals = 0;

            // Bundled to keep the kernel compact — unrolled in source for clarity.
            #define ACCUM(VAL, WGT)                                                  \
                if ((VAL) >= dummy) {                                                \
                    bool matched = false;                                            \
                    for (int k = 0; k < nvals; ++k) {                                \
                        if (fabs(val_list[k] - (VAL)) < 1e-5f) {                     \
                            weight_list[k] += (WGT);                                 \
                            matched = true;                                          \
                            break;                                                   \
                        }                                                            \
                    }                                                                \
                    if (!matched && nvals < 8) {                                     \
                        val_list[nvals] = (VAL);                                     \
                        weight_list[nvals] = (WGT);                                  \
                        nvals++;                                                     \
                    }                                                                \
                }

            ACCUM(v0, tnw)
            ACCUM(v1, tne)
            ACCUM(v2, tsw)
            ACCUM(v3, tse)
            ACCUM(v4, bnw)
            ACCUM(v5, bne)
            ACCUM(v6, bsw)
            ACCUM(v7, bse)
            #undef ACCUM

            float best_v;
            float best_w;
            if (nvals > 0) {
                best_v = val_list[0];
                best_w = weight_list[0];
                for (int k = 1; k < nvals; ++k) {
                    if (weight_list[k] > best_w) {
                        best_w = weight_list[k];
                        best_v = val_list[k];
                    }
                }
            } else {
                best_v = dummy;
                best_w = 0.0f;
            }

            const int out_off = batch_out_off + c * out_sC + w_out + Wo * (h_out + Ho * d_out);
            output_labels[out_off] = best_v;
            if (return_weight) {
                output_weights[out_off] = best_w;
            }
        }
    }
}
)METAL";

} // namespace fireants_fused_ops::metal
