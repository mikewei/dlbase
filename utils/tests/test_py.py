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
    
    # 测试使用默认参数
    @my_decorator
    def test_func3():
        return "test3"
    assert test_func3() == "test3-None-None"

    # 测试只传递部分参数
    @my_decorator(param1="only_one")
    def test_func4():
        return "test4"
    assert test_func4() == "test4-only_one-None"
    
def test_save_params():
    # 创建一个测试类
    class TestClass:
        __annotations__ = {'x': int}
        @save_params
        def __init__(self, x, y=10, **kwargs):
            self.initialized = True
            
        @save_params
        def method_with_args(self, a, b, c=None):
            return a + b
            
        @save_params
        def method_with_kwargs(self, **kwargs):
            return sum(kwargs.values())

        @save_params(True)
        def method_with_kwargs2(self, d, *args, **kwargs):
            return sum(d.values()) + sum(args) + sum(kwargs.values())
    
    TestClass.__annotations__ = {"x": int, "y": str}
    # 测试构造函数参数的保存
    obj = TestClass(1, y=20, extra="test")
    assert obj.initialized  # 确认__init__正常执行
    assert obj.x == 1  # type: ignore 确认位置参数被保存
    assert obj.y == 20  # type: ignore 确认关键字参数被保存
    assert obj.kwargs['extra'] == "test"  # type: ignore 确认额外的kwargs被保存
    # 测试使用默认值
    obj2 = TestClass(5)
    assert obj2.x == 5   # type: ignore
    assert obj2.y == 10  # type: ignore 使用默认值
    # 测试位置参数的保存
    result = obj.method_with_args(1, 2)
    assert result == 3  # 确认方法正常执行
    assert obj.a == 1  # type: ignore 确认参数被保存为属性
    assert obj.b == 2  # type: ignore
    assert obj.c is None  # type: ignore
    # 测试带默认值的参数
    obj.method_with_args(3, 4, c=5)
    assert obj.a == 3  # type: ignore
    assert obj.b == 4  # type: ignore
    assert obj.c == 5  # type: ignore
    # 测试关键字参数
    obj2 = TestClass(0)
    result = obj2.method_with_kwargs(x=1, y=2, z=3)
    assert result == 6  # 确认方法正常执行
    assert obj2.kwargs['x'] == 1  # type: ignore
    assert obj2.kwargs['y'] == 2  # type: ignore
    assert obj2.kwargs['z'] == 3  # type: ignore
    # 测试关键字参数展开
    obj3 = TestClass(0)
    result = obj3.method_with_kwargs2({'a': 100}, 10, x=1, y=2, z=3)
    assert result == 116  # 确认方法正常执行
    assert obj3.a == 100  # type: ignore
    assert obj3.d == {'a': 100}  # type: ignore
    assert obj3.args == (10,)  # type: ignore
    assert obj3.x == 1  # type: ignore
    assert obj3.y == 2  # type: ignore
    assert obj3.z == 3  # type: ignore
    assert obj3.kwargs == {'x': 1, 'y': 2, 'z': 3}  # type: ignore
