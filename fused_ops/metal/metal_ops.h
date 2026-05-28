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

// Declarations for Metal/MPS-only entry points exposed by src.cpp's pybind
// module. The CUDA build does not include these; the macOS build defines
// FIREANTS_FUSED_OPS_HAS_METAL so src.cpp pulls them in.

#pragma once

#include <torch/torch.h>
#include <vector>

// Backward pass of torch.nn.functional.grid_sample for 5D input with
// mode='bilinear', padding_mode='zeros'. Returns (grad_input, grad_grid).
// Forward is left to the native MPS F.grid_sample.
//
//   grad_output : [N, C, D_out, H_out, W_out], float32, contiguous, MPS
//   input       : [N, C, D_in,  H_in,  W_in],  float32, contiguous, MPS
//   grid        : [N, D_out, H_out, W_out, 3], float32, contiguous, MPS
//   align_corners : bool, must match the forward call
std::vector<torch::Tensor> grid_sample_3d_backward_mps(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& grid,
    bool align_corners);

// Forward pass of the FireANTs fused generic-label 3D grid sampler on MPS.
// Mirrors the CUDA path in FusedGridSamplerGenericLabel.cu — float32 only.
// Signature matches the CUDA `fused_grid_sampler_3d_generic_label_forward_impl`
// so it can be bound under the same Python name.
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
fused_grid_sampler_3d_generic_label_forward_mps(
    const torch::Tensor& input,
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights,
    const int64_t out_D,
    const int64_t out_H,
    const int64_t out_W,
    const float grid_xmin,
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax,
    const bool is_displacement,
    int64_t padding_mode,
    bool align_corners,
    bool return_weight,
    const std::optional<float> background_label = std::nullopt);
