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

// Metal/MPS implementation of adam_update_fused.
//
// Linker-level counterpart to fused_ops/src/AdamUtils.cu: on macOS the build
// compiles this .mm in place of the .cu, exposing the same C symbol so
// src.cpp's pybind11 binding does not need any platform branching.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MetalShaderLibrary.h>

#include "shaders/adam_kernel_msl.h"

namespace {

// Lazy-initialized so the library is compiled only on first use, not at
// module import time. Compilation is cached inside MetalShaderLibrary.
at::native::mps::MetalShaderLibrary& adam_library() {
    static at::native::mps::DynamicMetalShaderLibrary lib(
        fireants_fused_ops::metal::adam_kernel_msl_src);
    return lib;
}

const char* kernel_name_for(at::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat:    return "adam_update_fused_kernel_float";
        case at::kHalf:     return "adam_update_fused_kernel_half";
        case at::kBFloat16: return "adam_update_fused_kernel_bfloat";
        default: return nullptr;
    }
}

} // namespace

void adam_update_fused(torch::Tensor &grad,
                       torch::Tensor exp_avg,
                       torch::Tensor exp_avg_sq,
                       float beta1,
                       float beta2,
                       float eps) {
    TORCH_CHECK(grad.device().is_mps(),       "grad must be an MPS tensor");
    TORCH_CHECK(exp_avg.device().is_mps(),    "exp_avg must be an MPS tensor");
    TORCH_CHECK(exp_avg_sq.device().is_mps(), "exp_avg_sq must be an MPS tensor");
    TORCH_CHECK(grad.is_contiguous() && exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(),
                "adam_update_fused inputs must be contiguous");
    TORCH_CHECK(grad.scalar_type() == exp_avg.scalar_type() &&
                grad.scalar_type() == exp_avg_sq.scalar_type(),
                "adam_update_fused inputs must share a scalar type");

    const char* fname = kernel_name_for(grad.scalar_type());
    TORCH_CHECK(fname != nullptr,
                "adam_update_fused (MPS) supports float32/float16/bfloat16; got ",
                grad.scalar_type());

    const uint64_t n_elements = static_cast<uint64_t>(grad.numel());
    if (n_elements == 0) return;

    auto* kernel = adam_library().getCachedKernelFunctionPtr(fname);
    TORCH_CHECK(kernel != nullptr,
                "Failed to load Metal kernel '", fname, "' for adam_update_fused");

    kernel->runCommandBlock([&] {
        kernel->startEncoding();
        kernel->setArg(0, grad);
        kernel->setArg(1, exp_avg);
        kernel->setArg(2, exp_avg_sq);
        kernel->setArg(3, beta1);
        kernel->setArg(4, beta2);
        kernel->setArg(5, eps);
        kernel->setArg(6, n_elements);
        kernel->dispatch(n_elements);
    });
}
