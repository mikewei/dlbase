import math

__all__ = ['ExpDecay']

class ExpDecay:
    def __init__(self, start: float, end: float, unit: float = 1.0):
        self.start = start
        self.end = end
        self.unit = unit
    def __call__(self, t: float) -> float:
        return self.end + (self.start - self.end) * math.exp(-1. * t / self.unit)