import pytest
import math
from ..decay import ExpDecay

def test_exp_decay():
    decay = ExpDecay(start=1.0, end=0.0, unit=1.0)
    
    # 测试关键点
    assert decay(0) == 1.0  # 起始点
    assert pytest.approx(decay(1.0)) == 1.0 * math.exp(-1.0)  # t = unit
    assert pytest.approx(decay(100.0)) == 0.0  # 趋近终值
    
    # 测试不同参数组合
    decay2 = ExpDecay(start=2.0, end=1.0, unit=2.0)
    t = 2.0
    expected = 1.0 + math.exp(-1.0)  # t/unit = 1.0
    assert pytest.approx(decay2(t)) == expected

    # 测试异常情况
    with pytest.raises(ZeroDivisionError):
        decay_zero = ExpDecay(start=1.0, end=0.0, unit=0.0)
        decay_zero(1.0)
