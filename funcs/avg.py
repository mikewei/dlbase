class EWMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = 0
    def __call__(self, value):
        self.value = self.value * self.alpha + value * (1 - self.alpha)
        return self.value