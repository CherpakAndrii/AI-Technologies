from abc import ABC, abstractmethod


class INeuralNetwork(ABC):
    @abstractmethod
    def train(self, train_x: list[tuple[float]], train_y: list[float], epochs: int, learning_rate: float) -> None:
        pass

    @abstractmethod
    def test(self, test_x: list[tuple[float]], test_y: list[float]) -> tuple[float, float]:
        pass

    @abstractmethod
    def predict(self, x: tuple[float]) -> float:
        pass
