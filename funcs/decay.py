import math

class ExpDecay:
    def __init__(self, start, end, unit):
        self.start = start
        self.end = end
        self.unit = unit
    def __call__(self, t):
        return self.end + (self.start - self.end) * math.exp(-1. * t / self.unit)