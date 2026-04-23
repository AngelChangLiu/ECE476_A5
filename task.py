from typing import Tuple, TypeAlias, TypedDict

import torch


input_t: TypeAlias = Tuple[torch.Tensor, int]
output_t: TypeAlias = torch.Tensor


class TestSpec(TypedDict):
    length: int
    num_channels: int
    num_bins: int
    seed: int
