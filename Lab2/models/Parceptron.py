from math import fabs

import numpy as np

from interfaces.INeuralNetwork import INeuralNetwork
from models.InputLayerNeuron import InputLayerNeuron
from models.Neuron import Neuron
from models.SigmoidActivationFunction import SigmoidActivationFunction
from models.Synapse import Synapse
from models.ThresholdActivationFunction import ThresholdActivationFunction


class Parceptron(INeuralNetwork):
    input_neurons: list[InputLayerNeuron]
    hidden_neurons: list[Neuron]
    output_neurons: list[Neuron]

    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int):
        self.input_neurons = [InputLayerNeuron() for _ in range(input_size)]
        self.hidden_neurons = [Neuron([Synapse(n, 0.5) for n in self.input_neurons], 0, ThresholdActivationFunction(0.75)) for _ in range(hidden_layer_size)]
        self.output_neurons = [Neuron([Synapse(self.hidden_neurons[i], 1)], 0, ThresholdActivationFunction(0.75)) for i in range(output_size)]

    def train(self, train_x: list[tuple[float]], train_y: list[tuple[float]], epochs: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            perfect_counter = 0
            for train_case, expected_result in zip(train_x, train_y):
                result = self.predict(train_case)
                if all(fabs(exp_result - real_result) < 0.25 for exp_result, real_result in zip(expected_result, result)):
                    perfect_counter += 1
                error = [(exp_result - real_result) for exp_result, real_result in zip(expected_result, result)]
                for n, err in zip(self.hidden_neurons, error):
                    n.error_propagation(err, learning_rate)

            print("Epoch", epoch, "done, train accuracy:", perfect_counter,  '/', len(train_x))

    def test(self, test_x: list[tuple[float]], test_y: list[tuple[float]]) -> tuple[float, float]:
        correct_ctr = 0
        processed_ctr = 0
        for test_case, expected_result in zip(test_x, test_y):
            processed_ctr += 1
            result = self.predict(test_case)
            if all(fabs(exp_result - real_result) < 0.25 for exp_result, real_result in zip(expected_result, result)):
                correct_ctr += 1

        return correct_ctr, processed_ctr

    def predict(self, x: tuple[float]) -> list[float]:
        for n, x_val in zip(self.input_neurons, x):
            n.accept_value(x_val)

        Parceptron.feed(self.hidden_neurons)
        Parceptron.feed(self.output_neurons)

        return [output_neuron.get_value() for output_neuron in self.output_neurons]

    @staticmethod
    @np.vectorize
    def feed(n: Neuron):
        n.feedforward()

    @staticmethod
    @np.vectorize(excluded=['error', 'learning_rate'])
    def fit(n: Neuron, error: float, learning_rate: float) -> None:
        n.error_propagation(error, learning_rate)
