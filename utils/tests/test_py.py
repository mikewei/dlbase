from ..py import paramed_decorator, save_params

def test_paramed_decorator():
    # 定义一个带参数的装饰器
    @paramed_decorator
    def my_decorator(func, param1=None, param2=None):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return f"{result}-{param1}-{param2}"
        return wrapper
    
    # 使用装饰器装饰测试函数
    @my_decorator(param1="hello", param2="world")
    def test_func():
        return "test"
    
    # 测试装饰器是否正确传递参数并修改返回值
    assert test_func() == "test-hello-world"
    
    # 测试使用默认参数
    @my_decorator()
    def test_func2():
        return "test2"
    
    assert test_func2() == "test2-None-None"
    
    # 测试只传递部分参数
    @my_decorator(param1="only_one")
    def test_func3():
        return "test3"
    
    assert test_func3() == "test3-only_one-None"
    
def test_save_params():
    # 创建一个测试类
    class TestClass:
        @save_params
        def __init__(self, x, y=10, **kwargs):
            self.initialized = True
            
        @save_params
        def method_with_args(self, a, b, c=None):
            return a + b
            
        @save_params
        def method_with_kwargs(self, **kwargs):
            return sum(kwargs.values())
    
    # 测试构造函数参数的保存
    obj = TestClass(1, y=20, extra="test")
    assert obj.initialized  # 确认__init__正常执行
    assert obj.x == 1  # 确认位置参数被保存
    assert obj.y == 20  # 确认关键字参数被保存
    assert obj.kwargs['extra'] == "test"  # 确认额外的kwargs被保存
    
    # 测试使用默认值
    obj2 = TestClass(5)
    assert obj2.x == 5
    assert obj2.y == 10  # 使用默认值
    
    # 测试位置参数的保存
    result = obj.method_with_args(1, 2)
    
    assert result == 3  # 确认方法正常执行
    assert obj.a == 1  # 确认参数被保存为属性
    assert obj.b == 2
    assert obj.c is None
    
    # 测试带默认值的参数
    obj.method_with_args(3, 4, c=5)
    assert obj.a == 3
    assert obj.b == 4
    assert obj.c == 5
    
    # 测试关键字参数
    obj2 = TestClass(0)
    result = obj2.method_with_kwargs(x=1, y=2, z=3)
    
    assert result == 6  # 确认方法正常执行
    assert obj2.kwargs['x'] == 1
    assert obj2.kwargs['y'] == 2
    assert obj2.kwargs['z'] == 3