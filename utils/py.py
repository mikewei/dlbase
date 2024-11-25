from functools import wraps
import inspect
import contextlib

__all__ = ['paramed_decorator', 'save_params', 'NopContextManager']

def paramed_decorator(inner_decorator):
    @wraps(inner_decorator)
    def decorator_wrapper(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return inner_decorator(args[0])
        else:
            def partial_binded_decorator(func):
                return inner_decorator(func, *args, **kwargs)
            return partial_binded_decorator
    return decorator_wrapper

def save_params(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # 获取方法的完整签名
        sig = inspect.signature(method)
        # 绑定传入的参数
        bound_args = sig.bind(self, *args, **kwargs)
        # 将默认值应用到未传递的参数
        bound_args.apply_defaults()
        # 将参数存储为对象的属性
        for name, value in bound_args.arguments.items():
            if name != 'self':  # 排除self本身
                setattr(self, name, value)
        return method(self, *args, **kwargs)
    return wrapper

class NopContextManager(contextlib.AbstractContextManager):
    def __exit__(self, exc_type, exc_value, traceback):
        return False
