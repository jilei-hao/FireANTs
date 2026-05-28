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
#include "shaders/grid_sample_3d_generic_label_msl.h"

namespace {

// Bit positions kept in sync with the shader source.
constexpr uint32_t FLAG_HAS_AFFINE        = 1u << 0;
constexpr uint32_t FLAG_HAS_GRID          = 1u << 1;
constexpr uint32_t FLAG_HAS_GRID_AFFINE   = 1u << 2;
constexpr uint32_t FLAG_IS_DISPLACEMENT   = 1u << 3;
constexpr uint32_t FLAG_ALIGN_CORNERS     = 1u << 4;
constexpr uint32_t FLAG_RETURN_WEIGHT     = 1u << 5;
constexpr uint32_t FLAG_BROADCAST_INPUT   = 1u << 6;
constexpr uint32_t FLAG_BROADCAST_AFFINE  = 1u << 7;
constexpr uint32_t FLAG_BROADCAST_GRID    = 1u << 8;
constexpr uint32_t FLAG_BROADCAST_GAFFINE = 1u << 9;

// Must match GridSampleGL3dParams in the shader source — including padding /
// alignment. Keep all ints contiguous to avoid implicit padding.
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
    int padding_mode;
    float dummy;
    uint32_t flags;
};

at::native::mps::MetalShaderLibrary& library() {
    static at::native::mps::DynamicMetalShaderLibrary lib(
        fireants_fused_ops::metal::grid_sample_3d_generic_label_forward_msl_src);
    return lib;
}

inline void check_mps_float_contig(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_mps(),     name, ": must be on MPS");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, ": only float32 is supported");
    TORCH_CHECK(t.is_contiguous(),       name, ": must be contiguous");
}

} // namespace

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
    const std::optional<float> background_label) {

    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    check_mps_float_contig(input, "input");
    TORCH_CHECK(grid.has_value() || affine_3d.has_value(),
                "one of grid or affine_3d must be provided");

    // Determine output spatial shape and validate optional tensor metadata.
    int64_t D, H, W;
    if (grid.has_value()) {
        const auto& g = grid.value();
        check_mps_float_contig(g, "grid");
        TORCH_CHECK(g.dim() == 5 && g.size(-1) == 3,
                    "grid must have shape [B, D, H, W, 3]");
        D = g.size(1); H = g.size(2); W = g.size(3);
    } else {
        D = out_D; H = out_H; W = out_W;
    }
    if (affine_3d.has_value()) {
        const auto& a = affine_3d.value();
        check_mps_float_contig(a, "affine_3d");
        TORCH_CHECK(a.dim() == 3 && a.size(1) == 3 && a.size(2) == 4,
                    "affine_3d must have shape [B, 3, 4]");
    }
    if (grid_affine.has_value()) {
        const auto& ga = grid_affine.value();
        check_mps_float_contig(ga, "grid_affine");
        TORCH_CHECK(ga.dim() == 3 && ga.size(1) == 3 && ga.size(2) == 3,
                    "grid_affine must have shape [B, 3, 3]");
    }

    int64_t batch_size_max = input.size(0);
    if (affine_3d.has_value())   batch_size_max = std::max<int64_t>(batch_size_max, affine_3d.value().size(0));
    if (grid.has_value())        batch_size_max = std::max<int64_t>(batch_size_max, grid.value().size(0));
    if (grid_affine.has_value()) batch_size_max = std::max<int64_t>(batch_size_max, grid_affine.value().size(0));

    bool broadcast_input = false, broadcast_affine = false,
         broadcast_grid = false, broadcast_grid_affine = false;
    if (batch_size_max > 1) {
        if (input.size(0) == 1) broadcast_input = true;
        else TORCH_CHECK(input.size(0) == batch_size_max, "input batch mismatch");
        if (affine_3d.has_value() && affine_3d.value().size(0) == 1) broadcast_affine = true;
        else if (affine_3d.has_value())
            TORCH_CHECK(affine_3d.value().size(0) == batch_size_max, "affine_3d batch mismatch");
        if (grid.has_value() && grid.value().size(0) == 1) broadcast_grid = true;
        else if (grid.has_value())
            TORCH_CHECK(grid.value().size(0) == batch_size_max, "grid batch mismatch");
        if (grid_affine.has_value() && grid_affine.value().size(0) == 1) broadcast_grid_affine = true;
        else if (grid_affine.has_value())
            TORCH_CHECK(grid_affine.value().size(0) == batch_size_max, "grid_affine batch mismatch");
    }

    const int64_t N  = batch_size_max;
    const int64_t C  = input.size(1);
    const int64_t Di = input.size(2);
    const int64_t Hi = input.size(3);
    const int64_t Wi = input.size(4);

    // dummy = background fill / "ignored" sentinel. CUDA uses input.min() when no
    // background_label is supplied; mirror that.
    float dummy_val;
    if (background_label.has_value()) {
        dummy_val = background_label.value();
    } else {
        dummy_val = input.min().item<float>();
    }

    if (!output_labels.has_value()) {
        output_labels.emplace(torch::full({N, C, D, H, W}, dummy_val, input.options()));
    } else {
        TORCH_CHECK(output_labels.value().sizes() == at::IntArrayRef({N, C, D, H, W}),
                    "output_labels shape must be [N, C, D, H, W]");
        check_mps_float_contig(output_labels.value(), "output_labels");
        output_labels.value().fill_(dummy_val);
    }

    std::optional<torch::Tensor> out_weights_opt;
    if (return_weight) {
        if (!output_weights.has_value()) {
            output_weights.emplace(torch::zeros({N, C, D, H, W}, input.options()));
        } else {
            TORCH_CHECK(output_weights.value().sizes() == output_labels.value().sizes(),
                        "output_weights shape must match output_labels");
            check_mps_float_contig(output_weights.value(), "output_weights");
            output_weights.value().zero_();
        }
        out_weights_opt = output_weights;
    }

    const int64_t total = N * D * H * W;
    if (total == 0) {
        return {output_labels.value(), out_weights_opt};
    }

    TORCH_CHECK(padding_mode >= 0 && padding_mode <= 2,
                "padding_mode must be 0 (zeros), 1 (border), or 2 (reflection)");

    uint32_t flags = 0;
    if (affine_3d.has_value())    flags |= FLAG_HAS_AFFINE;
    if (grid.has_value())         flags |= FLAG_HAS_GRID;
    if (grid_affine.has_value())  flags |= FLAG_HAS_GRID_AFFINE;
    if (is_displacement)          flags |= FLAG_IS_DISPLACEMENT;
    if (align_corners)            flags |= FLAG_ALIGN_CORNERS;
    if (return_weight)            flags |= FLAG_RETURN_WEIGHT;
    if (broadcast_input)          flags |= FLAG_BROADCAST_INPUT;
    if (broadcast_affine)         flags |= FLAG_BROADCAST_AFFINE;
    if (broadcast_grid)           flags |= FLAG_BROADCAST_GRID;
    if (broadcast_grid_affine)    flags |= FLAG_BROADCAST_GAFFINE;

    GridSampleGL3dParams P{
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(Di), static_cast<int>(Hi), static_cast<int>(Wi),
        static_cast<int>(D),  static_cast<int>(H),  static_cast<int>(W),
        grid_xmin, grid_ymin, grid_zmin,
        grid_xmax, grid_ymax, grid_zmax,
        static_cast<int>(padding_mode),
        dummy_val,
        flags,
    };

    // The shader expects valid pointers for the optional buffers; bind zero-sized
    // placeholders when the corresponding optional is empty. The shader gates
    // the read via the matching FLAG_HAS_* bit so the placeholders are never
    // dereferenced.
    auto placeholder = torch::empty({1}, input.options());

    const auto& aff_buf  = affine_3d.has_value()    ? affine_3d.value()    : placeholder;
    const auto& grd_buf  = grid.has_value()         ? grid.value()         : placeholder;
    const auto& gaff_buf = grid_affine.has_value()  ? grid_affine.value()  : placeholder;
    const auto& wgt_buf  = return_weight            ? output_weights.value() : placeholder;

    auto* kernel = library().getCachedKernelFunctionPtr("grid_sample_3d_generic_label_forward");
    TORCH_CHECK(kernel != nullptr,
                "fused_grid_sampler_3d_generic_label_forward_mps: kernel not found");

    kernel->runCommandBlock([&] {
        kernel->startEncoding();
        kernel->setArg(0, input);
        kernel->setArg(1, aff_buf);
        kernel->setArg(2, grd_buf);
        kernel->setArg(3, gaff_buf);
        kernel->setArg(4, output_labels.value());
        kernel->setArg(5, wgt_buf);
        kernel->setArg(6, P);
        kernel->dispatch(static_cast<uint64_t>(total));
    });

    return {output_labels.value(), out_weights_opt};
}
