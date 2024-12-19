__all__ = ['EWMA', 'AccuAvg']

class EWMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = 0
    def __call__(self, value):
        self.value = self.value * self.alpha + value * (1 - self.alpha)
        return self.value

class AccuAvg:
    def __init__(self):
        self.values = [0]
        self.count = 0
    def add(self, *values, count=1):
        if len(values) > len(self.values):
            self.values.extend([0] * (len(values) - len(self.values)))
        for i in range(len(values)):
            self.values[i] += values[i]
        self.count += count
    def get(self):
        if len(self.values) == 1:
            return self.values[0] / self.count
        else:
            return [v / self.count for v in self.values]
    def __call__(self, *values):
        self.add(*values)
        return self.get()
