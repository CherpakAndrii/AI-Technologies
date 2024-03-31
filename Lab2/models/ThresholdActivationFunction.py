from interfaces.IActivationFunction import IActivationFunction


class ThresholdActivationFunction(IActivationFunction):
    threshold: float

    def __init__(self, threshold: float):
        self.threshold = threshold

    def compute(self, inputs_sum: float) -> float:
        return 1 if inputs_sum >= self.threshold else 0
