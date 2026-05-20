# Copyright (c) 2026 Rohit Jena. All rights reserved.
#
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE


import os
import sys

from setuptools import setup
from torch.utils import cpp_extension
import torch

here = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(here, 'include')
metal_dir = os.path.join(here, 'metal')

is_macos = sys.platform == 'darwin'
has_cuda = torch.version.cuda is not None and not is_macos


def build_cuda_extension():
    return cpp_extension.CUDAExtension(
        name='fireants_fused_ops',
        sources=[
            'src/src.cpp',
            'src/CrossCorrelation.cu',
            'src/FusedGridSampler.cu',
            'src/FusedGridSamplerGenericLabel.cu',
            'src/FusedGridComposer.cu',
            'src/FusedGenerateGrid.cu',
            'src/AdamUtils.cu',
            'src/GaussianBlurFFT.cu',
            'src/MutualInformation.cu',
        ],
        include_dirs=[include_dir] + torch.utils.cpp_extension.include_paths(),
        library_dirs=torch.utils.cpp_extension.library_paths(),
        define_macros=[('FIREANTS_FUSED_OPS_HAS_CUDA', '1')],
        extra_compile_args={
            'nvcc': ['-lineinfo'],
        },
    )


def build_macos_metal_extension():
    # CppExtension on macOS: link Metal + Foundation, compile the ObjC++ host
    # wrappers under metal/. CUDA-only kernels are not built; src.cpp gates
    # their pybind binders on FIREANTS_FUSED_OPS_HAS_CUDA.
    ext = cpp_extension.CppExtension(
        name='fireants_fused_ops',
        sources=[
            'src/src.cpp',
            'metal/AdamUtils.mm',
        ],
        include_dirs=[include_dir, metal_dir] + torch.utils.cpp_extension.include_paths(),
        library_dirs=torch.utils.cpp_extension.library_paths(),
        define_macros=[('FIREANTS_FUSED_OPS_HAS_METAL', '1')],
        extra_compile_args={
            'cxx': [
                '-std=c++17',
                '-fobjc-arc',
            ],
        },
        extra_link_args=[
            '-framework', 'Metal',
            '-framework', 'Foundation',
            '-framework', 'MetalPerformanceShaders',
            '-framework', 'MetalPerformanceShadersGraph',
            # cpp_extension's library_paths() supplies -L but not rpath on macOS;
            # add it explicitly so libc10 / libtorch resolve at import time.
            *sum(
                (['-Wl,-rpath,' + p] for p in torch.utils.cpp_extension.library_paths()),
                [],
            ),
        ],
    )
    return ext


def _register_mm_extension():
    # setuptools' UnixCCompiler defaults do not include .mm; register it so
    # build_ext routes the file to the C++ compiler (clang infers Objective-C++
    # from the .mm extension on its own).
    for mod_name in ('setuptools._distutils.unixccompiler', 'distutils.unixccompiler'):
        try:
            mod = __import__(mod_name, fromlist=['UnixCCompiler'])
        except ImportError:
            continue
        cls = getattr(mod, 'UnixCCompiler', None)
        if cls is None:
            continue
        exts = list(getattr(cls, 'src_extensions', None) or [])
        if '.mm' not in exts:
            cls.src_extensions = exts + ['.mm']
        lang_map = getattr(cls, 'language_map', None)
        if isinstance(lang_map, dict):
            lang_map.setdefault('.mm', 'c++')


if is_macos:
    _register_mm_extension()
    ext_modules = [build_macos_metal_extension()]
elif has_cuda:
    ext_modules = [build_cuda_extension()]
else:
    raise RuntimeError(
        "fireants_fused_ops requires either CUDA (Linux) or Metal (macOS). "
        "Neither was detected."
    )

setup(
    name='fireants_fused_ops',
    version='1.0.0',
    description='Fused operations for FireANTs (CUDA and Metal backends)',
    author='Rohit Jena',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch>=2.3.0'],
)
