#include <torch/extension.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iostream>

namespace {

__global__ void histogram_shared_kernel(
    const uint8_t* __restrict__ data,
    int32_t* __restrict__ histogram,
    int length,
    int num_channels,
    int num_bins) {
    
    // ECE 476 TODO
    // If you'd like to implement with pure CUDA, write your kernel here.
    // Otherwise you can leave it empty.
}

}  // namespace

torch::Tensor histogram_kernel(torch::Tensor data, int64_t num_bins) {
  TORCH_CHECK(data.is_cuda(), "data must be a CUDA tensor");
  TORCH_CHECK(data.scalar_type() == torch::kUInt8, "data must have dtype torch.uint8");
  TORCH_CHECK(data.dim() == 2, "data must have shape [length, num_channels]");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(num_bins > 0 && num_bins <= 256, "num_bins must be in the range [1, 256]");

  std::cout << "[histogram_kernel] Entered C++ histogram_kernel, input is valid." << std::endl;

  const int length = static_cast<int>(data.size(0));
  const int num_channels = static_cast<int>(data.size(1));
  const int bins = static_cast<int>(num_bins);

  auto histogram = torch::zeros(
      {num_channels, bins},
      torch::TensorOptions().dtype(torch::kInt32).device(data.device()));
  
  // ECE 476 TODO
  // If you are implementing in pure CUDA
  // Set up and launch your kernel here.

  return histogram;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("histogram_kernel", &histogram_kernel, "Multi-channel histogram kernel");
}
