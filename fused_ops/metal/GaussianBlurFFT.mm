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

#include "shaders/gaussian_blur_fft_msl.h"

namespace {

struct GaussianBlurFFT2Params {
    int H, W, total;
    int ys, xs;
    float multiplier;
};

struct GaussianBlurFFT3Params {
    int D, H, W, total;
    int zs, ys, xs;
    float multiplier;
};

at::native::mps::MetalShaderLibrary& library() {
    static at::native::mps::DynamicMetalShaderLibrary lib(
        fireants_fused_ops::metal::gaussian_blur_fft_msl_src);
    return lib;
}

} // namespace

// Mirrors fused_ops/src/GaussianBlurFFT.cu::gaussian_blur_fft2.
// Operates in place on a complex64 tensor of shape [N, C, H, W].
void gaussian_blur_fft2(torch::Tensor& im_fft,
                        int64_t ys, int64_t xs, int64_t ye, int64_t xe,
                        float multiplier) {
    (void)ye; (void)xe;   // CUDA kernel ignores ye/xe; preserved for ABI compat
    TORCH_CHECK(im_fft.device().is_mps(), "gaussian_blur_fft2: im_fft must be MPS");
    TORCH_CHECK(im_fft.is_contiguous(), "gaussian_blur_fft2: im_fft must be contiguous");
    TORCH_CHECK(im_fft.scalar_type() == at::kComplexFloat,
                "gaussian_blur_fft2: only complex64 supported (got ", im_fft.scalar_type(), ")");
    TORCH_CHECK(im_fft.dim() == 4, "gaussian_blur_fft2: expected 4D tensor [N,C,H,W]");

    const int N = static_cast<int>(im_fft.size(0));
    const int C = static_cast<int>(im_fft.size(1));
    const int H = static_cast<int>(im_fft.size(2));
    const int W = static_cast<int>(im_fft.size(3));
    const int total = N * C * H * W;
    if (total == 0) return;

    GaussianBlurFFT2Params P{H, W, total,
                             static_cast<int>(ys), static_cast<int>(xs),
                             multiplier};

    auto* kernel = library().getCachedKernelFunctionPtr("gaussian_blur_fft2");
    TORCH_CHECK(kernel != nullptr, "gaussian_blur_fft2: failed to load Metal kernel");

    kernel->runCommandBlock([&] {
        kernel->startEncoding();
        kernel->setArg(0, im_fft);
        kernel->setArg(1, P);
        kernel->dispatch(static_cast<uint64_t>(total));
    });
}

// Mirrors fused_ops/src/GaussianBlurFFT.cu::gaussian_blur_fft3.
// Operates in place on a complex64 tensor of shape [N, C, D, H, W].
void gaussian_blur_fft3(torch::Tensor& im_fft,
                        int64_t zs, int64_t ys, int64_t xs,
                        int64_t ze, int64_t ye, int64_t xe,
                        float multiplier) {
    (void)ze; (void)ye; (void)xe;
    TORCH_CHECK(im_fft.device().is_mps(), "gaussian_blur_fft3: im_fft must be MPS");
    TORCH_CHECK(im_fft.is_contiguous(), "gaussian_blur_fft3: im_fft must be contiguous");
    TORCH_CHECK(im_fft.scalar_type() == at::kComplexFloat,
                "gaussian_blur_fft3: only complex64 supported (got ", im_fft.scalar_type(), ")");
    TORCH_CHECK(im_fft.dim() == 5, "gaussian_blur_fft3: expected 5D tensor [N,C,D,H,W]");

    const int N = static_cast<int>(im_fft.size(0));
    const int C = static_cast<int>(im_fft.size(1));
    const int D = static_cast<int>(im_fft.size(2));
    const int H = static_cast<int>(im_fft.size(3));
    const int W = static_cast<int>(im_fft.size(4));
    const int total = N * C * D * H * W;
    if (total == 0) return;

    GaussianBlurFFT3Params P{D, H, W, total,
                             static_cast<int>(zs), static_cast<int>(ys), static_cast<int>(xs),
                             multiplier};

    auto* kernel = library().getCachedKernelFunctionPtr("gaussian_blur_fft3");
    TORCH_CHECK(kernel != nullptr, "gaussian_blur_fft3: failed to load Metal kernel");

    kernel->runCommandBlock([&] {
        kernel->startEncoding();
        kernel->setArg(0, im_fft);
        kernel->setArg(1, P);
        kernel->dispatch(static_cast<uint64_t>(total));
    });
}
