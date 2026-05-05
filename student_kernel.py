from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load

from task import input_t, output_t


HERE = Path(__file__).resolve().parent
CUDA_MODULE = None


def _load_cuda_module():
    global CUDA_MODULE
    if CUDA_MODULE is None:
        CUDA_MODULE = load(
            name="histogram_cuda",
            sources=[str(HERE / "histogram.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=bool(os.getenv("VERBOSE_COMPILE")),
        )
    return CUDA_MODULE


def preload_custom_kernel() -> None:
    """Build and load any kernel resources needed by custom_kernel."""
    _load_cuda_module()


def custom_kernel(data: input_t) -> output_t:
    """Student entry point.

    The starter implementation calls the CUDA extension in histogram.cu. You may
    replace this body with Triton, TileLang, torch operators, or another local
    implementation as long as it returns a [num_channels, num_bins] int32 tensor.
    """
    #
    # ECE 476 TODO
    # This is the entry function we'll be calling.
    # The current example calls into the CUDA implementation
    # But you can implement it however you'd like.
    #
    # You are allowed to import new modules, etc.
    #

    # array, num_bins = data
    # if not array.is_cuda:
    #     array = array.cuda()

    # array = array.contiguous()
    # return _load_cuda_module().histogram_kernel(array, int(num_bins))


    array, num_bins = data

    # move input to GPU if needed
    if not array.is_cuda:
        array = array.cuda()

    # keep memory contiguous for CUDA access
    array = array.contiguous()

    # call the CUDA implementation in histogram.cu
    return _load_cuda_module().histogram_kernel(array, int(num_bins))
