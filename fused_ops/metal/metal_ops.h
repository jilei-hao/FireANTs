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
