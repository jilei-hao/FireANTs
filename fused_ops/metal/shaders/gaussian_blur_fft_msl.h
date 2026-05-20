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

// In-place gaussian multiplier in FFT space, mirroring
// fused_ops/src/GaussianBlurFFT.cu::gaussian_blur_fft{2,3}_kernel.
//
// The host code passes a torch complex64 tensor directly; complex64 is laid
// out as alternating (real, imag) floats, so the shader views the buffer as
// float2 (xy = real, imag). The gaussian multiplier is real-valued, so each
// component is scaled identically.
//
// Math (per element at spatial offset (z, y, x) into the cropped FFT view):
//   sigma_d   = dim_size_d / 4
//   freq_d    = (start_d + idx_d) / sigma_d
//   weight    = exp(-0.5 * sum_d freq_d^2) * multiplier
//   z[i]     *= weight    (in place, applied independently to re and im)
// dim_size_d is the cropped extent along axis d, start_d is the frequency
// offset of the first sample in the crop (typically negative).

namespace fireants_fused_ops::metal {

static const char* gaussian_blur_fft_msl_src = R"METAL(
#include <metal_stdlib>
using namespace metal;

// 2D variant.
//   im_fft : N x C x H x W float2 (=> complex64)
struct GaussianBlurFFT2Params {
    int H;
    int W;
    int total;          // N * C * H * W
    int ys;             // frequency start along H
    int xs;             // frequency start along W
    float multiplier;
};

kernel void gaussian_blur_fft2(
    device float2*                  im_fft  [[ buffer(0) ]],
    constant GaussianBlurFFT2Params& P      [[ buffer(1) ]],
    uint                            tid     [[ thread_position_in_grid ]],
    uint                            gsize   [[ threads_per_grid ]])
{
    const float sigma_y = float(P.H) / 4.0f;
    const float sigma_x = float(P.W) / 4.0f;
    for (int i = int(tid); i < P.total; i += int(gsize)) {
        int w = i % P.W;
        int h = (i / P.W) % P.H;
        float yf = float(P.ys + h) / sigma_y;
        float xf = float(P.xs + w) / sigma_x;
        float weight = exp(-0.5f * (yf * yf + xf * xf)) * P.multiplier;
        float2 v = im_fft[i];
        im_fft[i] = float2(v.x * weight, v.y * weight);
    }
}

// 3D variant.
//   im_fft : N x C x D x H x W float2
struct GaussianBlurFFT3Params {
    int D;
    int H;
    int W;
    int total;          // N * C * D * H * W
    int zs;
    int ys;
    int xs;
    float multiplier;
};

kernel void gaussian_blur_fft3(
    device float2*                  im_fft  [[ buffer(0) ]],
    constant GaussianBlurFFT3Params& P      [[ buffer(1) ]],
    uint                            tid     [[ thread_position_in_grid ]],
    uint                            gsize   [[ threads_per_grid ]])
{
    const float sigma_z = float(P.D) / 4.0f;
    const float sigma_y = float(P.H) / 4.0f;
    const float sigma_x = float(P.W) / 4.0f;
    const int HW = P.H * P.W;
    for (int i = int(tid); i < P.total; i += int(gsize)) {
        int w = i % P.W;
        int h = (i / P.W) % P.H;
        int d = (i / HW) % P.D;
        float zf = float(P.zs + d) / sigma_z;
        float yf = float(P.ys + h) / sigma_y;
        float xf = float(P.xs + w) / sigma_x;
        float weight = exp(-0.5f * (xf * xf + yf * yf + zf * zf)) * P.multiplier;
        float2 v = im_fft[i];
        im_fft[i] = float2(v.x * weight, v.y * weight);
    }
}
)METAL";

} // namespace fireants_fused_ops::metal
