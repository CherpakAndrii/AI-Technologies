from abc import ABC, abstractmethod


class IActivationFunction(ABC):
    @abstractmethod
    def compute(self, inputs_sum: float) -> float:
        pass

    @abstractmethod
    def derivative(self, output: float):
        pass
