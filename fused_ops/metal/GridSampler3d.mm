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

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MetalShaderLibrary.h>

#include "metal_ops.h"
#include "shaders/grid_sample_3d_msl.h"

namespace {

// POD parameter block mirroring the GridSample3dParams struct in the shader.
// Lay-out must match exactly.
struct GridSample3dParams {
    int N;
    int C;
    int D_in;
    int H_in;
    int W_in;
    int D_out;
    int H_out;
    int W_out;
    int align_corners;
};

at::native::mps::MetalShaderLibrary& library() {
    static at::native::mps::DynamicMetalShaderLibrary lib(
        fireants_fused_ops::metal::grid_sample_3d_backward_msl_src);
    return lib;
}

} // namespace

std::vector<torch::Tensor> grid_sample_3d_backward_mps(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& grid,
    bool align_corners)
{
    TORCH_CHECK(grad_output.device().is_mps() && input.device().is_mps() && grid.device().is_mps(),
                "grid_sample_3d_backward_mps: all inputs must be MPS tensors");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous() && grid.is_contiguous(),
                "grid_sample_3d_backward_mps: all inputs must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat &&
                input.scalar_type() == at::kFloat &&
                grid.scalar_type() == at::kFloat,
                "grid_sample_3d_backward_mps: only float32 supported");
    TORCH_CHECK(input.dim() == 5,
                "grid_sample_3d_backward_mps: input must be 5D [N,C,D,H,W]; got dim ", input.dim());
    TORCH_CHECK(grid.dim() == 5 && grid.size(-1) == 3,
                "grid_sample_3d_backward_mps: grid must be 5D [N,D,H,W,3]");
    TORCH_CHECK(grad_output.dim() == 5,
                "grid_sample_3d_backward_mps: grad_output must be 5D");

    const int N      = static_cast<int>(input.size(0));
    const int C      = static_cast<int>(input.size(1));
    const int D_in   = static_cast<int>(input.size(2));
    const int H_in   = static_cast<int>(input.size(3));
    const int W_in   = static_cast<int>(input.size(4));
    const int D_out  = static_cast<int>(grid.size(1));
    const int H_out  = static_cast<int>(grid.size(2));
    const int W_out  = static_cast<int>(grid.size(3));

    TORCH_CHECK(grad_output.size(0) == N && grad_output.size(1) == C &&
                grad_output.size(2) == D_out && grad_output.size(3) == H_out &&
                grad_output.size(4) == W_out,
                "grid_sample_3d_backward_mps: grad_output shape mismatch");
    TORCH_CHECK(grid.size(0) == N,
                "grid_sample_3d_backward_mps: grid batch size mismatch");

    auto grad_input = torch::zeros_like(input);
    auto grad_grid  = torch::empty_like(grid);

    const uint64_t total_threads =
        static_cast<uint64_t>(N) * D_out * H_out * W_out;
    if (total_threads == 0) {
        return {grad_input, grad_grid};
    }

    GridSample3dParams params{
        N, C, D_in, H_in, W_in, D_out, H_out, W_out,
        align_corners ? 1 : 0
    };

    auto* kernel = library().getCachedKernelFunctionPtr("grid_sample_3d_backward");
    TORCH_CHECK(kernel != nullptr,
                "grid_sample_3d_backward_mps: failed to load Metal kernel");

    kernel->runCommandBlock([&] {
        kernel->startEncoding();
        kernel->setArg(0, grad_output);
        kernel->setArg(1, input);
        kernel->setArg(2, grid);
        kernel->setArg(3, grad_input);
        kernel->setArg(4, grad_grid);
        kernel->setArg(5, params);
        kernel->dispatch(total_threads);
    });

    return {grad_input, grad_grid};
}
