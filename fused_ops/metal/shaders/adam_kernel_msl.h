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

// Metal Shading Language source for the fused Adam update kernel.
// Compiled at first use by at::native::mps::MetalShaderLibrary; kept inline
// (rather than as a .metal file built with xcrun metal) so the extension
// builds on macOS without the optional Metal Toolchain component.
//
// Mirrors fused_ops/src/AdamUtils.cu::adam_update_fused_kernel:
//   grad[i] = exp_avg[i] / beta1 / (sqrt(exp_avg_sq[i] / beta2) + eps)

namespace fireants_fused_ops::metal {

static const char* adam_kernel_msl_src = R"METAL(
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void adam_update_fused_kernel(
    device T*               grad        [[ buffer(0) ]],
    device const T*         exp_avg     [[ buffer(1) ]],
    device const T*         exp_avg_sq  [[ buffer(2) ]],
    constant float&         beta1       [[ buffer(3) ]],
    constant float&         beta2       [[ buffer(4) ]],
    constant float&         eps         [[ buffer(5) ]],
    constant ulong&         n_elements  [[ buffer(6) ]],
    uint                    tid         [[ thread_position_in_grid ]],
    uint                    gsize       [[ threads_per_grid ]])
{
    // Math in float to match the CUDA path's opmath promotion for half/bf16.
    for (ulong i = ulong(tid); i < n_elements; i += ulong(gsize)) {
        float ea  = float(exp_avg[i]);
        float eas = float(exp_avg_sq[i]);
        float out = (ea / beta1) / (sqrt(eas / beta2) + eps);
        grad[i] = T(out);
    }
}

template
[[host_name("adam_update_fused_kernel_float")]]
kernel void adam_update_fused_kernel<float>(
    device float*, device const float*, device const float*,
    constant float&, constant float&, constant float&, constant ulong&,
    uint, uint);

template
[[host_name("adam_update_fused_kernel_half")]]
kernel void adam_update_fused_kernel<half>(
    device half*, device const half*, device const half*,
    constant float&, constant float&, constant float&, constant ulong&,
    uint, uint);

template
[[host_name("adam_update_fused_kernel_bfloat")]]
kernel void adam_update_fused_kernel<bfloat>(
    device bfloat*, device const bfloat*, device const bfloat*,
    constant float&, constant float&, constant float&, constant ulong&,
    uint, uint);
)METAL";

} // namespace fireants_fused_ops::metal
