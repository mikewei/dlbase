import pytest
from ..avg import EWMA

def test_ewma():
    # 初始化测试
    ewma = EWMA(alpha=0.9)
    assert ewma.alpha == 0.9
    assert ewma.value == 0
    
    # 计算测试
    # 第一次更新: 0 * 0.9 + 10 * 0.1 = 1.0
    result1 = ewma(10)
    assert result1 == pytest.approx(1.0)
    
    # 第二次更新: 1.0 * 0.9 + 20 * 0.1 = 2.9
    result2 = ewma(20)
    assert result2 == pytest.approx(2.9) 