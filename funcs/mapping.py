import torch
from torch import Tensor
from typing import Optional, Tuple

def rescale_tensor(x: Tensor, dst_range: Tuple[float, float], src_range: Tuple[Optional[float], Optional[float]] = (None, None)) -> Tensor:
    min_dst, max_dst = dst_range
    assert min_dst < max_dst
    if src_range != (None, None):
        x = x.clamp(*src_range)
    min_src = float(x.min().item()) if src_range[0] is None else src_range[0]
    max_src = float(x.max().item()) if src_range[1] is None else src_range[1]
    assert min_src < max_src
    return (x - min_src) / (max_src - min_src) * (max_dst - min_dst) + min_dst
