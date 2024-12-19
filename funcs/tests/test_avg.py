import pytest
from ..avg import EWMA, AccuAvg

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

def test_accu_avg():
    # 初始化测试
    avg = AccuAvg()
    assert avg.count == 0
    assert avg.values == [0]
    
    # 单值测试
    avg.add(10)
    assert avg.get() == 10
    
    # 多值测试
    avg = AccuAvg()
    avg.add(10, 20)
    assert avg.get() == [10, 20]
    
    # 累积测试
    avg.add(20, 40)
    assert avg.get() == [15, 30]  # (10+20)/2, (20+40)/2
    
    # count参数测试
    avg = AccuAvg()
    avg.add(10, count=2)
    assert avg.get() == 5
    
    # __call__测试
    avg = AccuAvg()
    avg.add(10, 20)
    result = avg(10, 30)
    assert result == [10, 25]
