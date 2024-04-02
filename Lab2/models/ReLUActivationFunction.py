from interfaces.IActivationFunction import IActivationFunction


class ReLUActivationFunction(IActivationFunction):
    def compute(self, inputs_sum: float) -> float:
        return 0 if inputs_sum < 0 else inputs_sum

    def derivative(self, output: float):
        raise NotImplemented
