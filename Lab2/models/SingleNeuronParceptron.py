import numpy as np

from interfaces.INeuralNetwork import INeuralNetwork
from models.InputLayerNeuron import InputLayerNeuron
from models.Neuron import Neuron
from models.Synapse import Synapse
from models.ThresholdActivationFunction import ThresholdActivationFunction


class SingleNeuronParceptron(INeuralNetwork):
    input_neurons: list[InputLayerNeuron]
    output_neurons: list[Neuron]

    def __init__(self, input_size: int, output_size: int):
        self.input_neurons = [InputLayerNeuron() for _ in range(input_size)]
        self.output_neurons = [Neuron([Synapse(n, 0.5) for n in self.input_neurons], 0, ThresholdActivationFunction(0.75)) for _ in range(output_size)]

    def train(self, train_x: list[tuple[float]], train_y: list[tuple[float]], epochs: int, learning_rate: float) -> None:
        for epoch in range(epochs):
            perfect_counter = 0
            for train_case, expected_result in zip(train_x, train_y):
                result = self.predict(train_case)
                if all(exp_result == real_result for exp_result, real_result in zip(expected_result, result)):
                    perfect_counter += 1
                error = [(exp_result - real_result) for exp_result, real_result in zip(expected_result, result)]
                for n, err in zip(self.output_neurons, error):
                    n.change_weights(err, learning_rate)

            print("Epoch", epoch, "done, train accuracy:", perfect_counter,  '/', len(train_x))

    def test(self, test_x: list[tuple[float]], test_y: list[tuple[float]]) -> tuple[float, float]:
        correct_ctr = 0
        processed_ctr = 0
        for test_case, expected_result in zip(test_x, test_y):
            processed_ctr += 1
            result = self.predict(test_case)
            if all(exp_result == real_result for exp_result, real_result in zip(expected_result, result)):
                correct_ctr += 1

        return correct_ctr, processed_ctr

    def predict(self, x: tuple[float]) -> list[float]:
        for n, x_val in zip(self.input_neurons, x):
            n.accept_value(x_val)

        for n in self.output_neurons:
            n.feedforward()
        return [n.get_value() for n in self.output_neurons]

