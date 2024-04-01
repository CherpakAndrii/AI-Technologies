import numpy as np

from interfaces.INeuralNetwork import INeuralNetwork
from models.InputLayerNeuron import InputLayerNeuron
from models.Neuron import Neuron
from models.Synapse import Synapse
from models.ThresholdActivationFunction import ThresholdActivationFunction


class Parceptron(INeuralNetwork):
    input_neurons: list[InputLayerNeuron]
    hidden_neurons: list[Neuron]
    output_neuron: Neuron

    def __init__(self, input_size: int, hidden_layer_size: int):
        self.input_neurons = [InputLayerNeuron() for _ in range(input_size)]
        self.hidden_neurons = [Neuron([Synapse(n, 0.5) for n in self.input_neurons], 0, ThresholdActivationFunction(0.75)) for _ in range(hidden_layer_size)]
        self.output_neuron = Neuron([Synapse(n, 1/hidden_layer_size) for n in self.hidden_neurons], 0, ThresholdActivationFunction(0.75))

    def train(self, train_x: list[tuple[float]], train_y: list[float], epochs: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            perfect_counter = 0
            for train_case, expected_result in zip(train_x, train_y):
                result = self.predict(train_case)
                if expected_result == result:
                    perfect_counter += 1
                error = expected_result - result
                Parceptron.fit(self.hidden_neurons, error, learning_rate)
                # self.output_neuron.change_weights(error, learning_rate)

            print("Epoch", epoch, "done, train accuracy:", perfect_counter,  '/', len(train_x))

    def test(self, test_x: list[tuple[float]], test_y: list[float]) -> tuple[float, float]:
        correct_ctr = 0
        processed_ctr = 0
        for test_case, expected_result in zip(test_x, test_y):
            processed_ctr += 1
            if self.predict(test_case) == expected_result:
                correct_ctr += 1

        return correct_ctr, processed_ctr

    def predict(self, x: tuple[float]) -> float:
        for n, x_val in zip(self.input_neurons, x):
            n.accept_value(x_val)

        Parceptron.feed(self.hidden_neurons)

        self.output_neuron.feedforward()
        return self.output_neuron.get_value()

    @staticmethod
    @np.vectorize
    def feed(n: Neuron):
        n.feedforward()

    @staticmethod
    @np.vectorize(excluded=['error', 'learning_rate'])
    def fit(n: Neuron, error: float, learning_rate: float) -> None:
        n.change_weights(error, learning_rate)
