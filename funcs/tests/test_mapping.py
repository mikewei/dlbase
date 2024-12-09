import pytest
import torch
from ..mapping import rescale_tensor

def test_rescale_tensor():
    # 创建测试数据
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 测试基本缩放功能
    dst_range = (0.0, 1.0)
    result = rescale_tensor(x, dst_range)
    assert torch.allclose(result, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    # 测试自定义目标范围
    dst_range = (-1.0, 1.0)
    result = rescale_tensor(x, dst_range)
    assert torch.allclose(result, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
    
    # 测试自定义源范围
    src_range = (0.0, 10.0)
    dst_range = (0.0, 1.0)
    result = rescale_tensor(x, dst_range, src_range)
    assert torch.allclose(result, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    
    # 测试数值截断
    x = torch.tensor([-1.0, 0.0, 5.0, 11.0])
    src_range = (0.0, 10.0)
    dst_range = (0.0, 1.0)
    result = rescale_tensor(x, dst_range, src_range)
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.5, 1.0]))
    
    # 测试异常情况
    with pytest.raises(AssertionError):
        rescale_tensor(x, (1.0, 0.0))  # 目标范围最小值大于最大值
