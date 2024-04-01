from math import exp

from interfaces.IActivationFunction import IActivationFunction


class SigmoidActivationFunction(IActivationFunction):
    def compute(self, inputs_sum: float) -> float:
        return 1 / (1 + exp(-inputs_sum))

    def derivative(self, output: float):
        return output * (1 - output)
