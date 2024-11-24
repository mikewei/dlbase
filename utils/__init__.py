import contextlib

class NopContextManager(contextlib.AbstractContextManager):
    def __exit__(self, exc_type, exc_value, traceback):
        return False