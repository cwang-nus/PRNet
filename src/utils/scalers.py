class Scaler():
    def __init__(self, maxValue):
        self.maxValue = maxValue

    def transform(self, x):
        return x / self.maxValue + 0.01

    def inverse_transform(self, x):
        return (x - 0.01) * self.maxValue
