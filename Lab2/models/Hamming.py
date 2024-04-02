from interfaces.INeuralNetwork import INeuralNetwork
from models.HammingNeuron import HammingNeuron
from models.InputLayerNeuron import InputLayerNeuron
from models.Neuron import Neuron
from models.ReLUActivationFunction import ReLUActivationFunction
from models.Synapse import Synapse


class Hamming(INeuralNetwork):
    input_neurons: list[InputLayerNeuron]
    hidden_neurons1: list[Neuron]
    hidden_neurons2: list[HammingNeuron]

    def __init__(self, input_size: int, output_size: int):
        self.input_neurons = [InputLayerNeuron() for _ in range(input_size)]
        self.hidden_neurons1 = [Neuron([], 0, ReLUActivationFunction())
                                for _ in range(output_size)]
        self.hidden_neurons2 = [HammingNeuron([], 0) for _ in range(output_size)]
        for i in range(output_size):
            self.hidden_neurons2[i].inputs.append(Synapse(self.hidden_neurons1[i], 1))
            for j in range(output_size):
                self.hidden_neurons2[i].inputs.append(Synapse(self.hidden_neurons2[j], 1 if i == j else -1/output_size))

    def train(self, train_x: list[tuple[float]], train_y: list[float], epochs: int = 10, learning_rate: float = 0.01) -> None:
        if len(train_x) != len(self.hidden_neurons1):
            raise ValueError("The length of train_x and hidden_neurons1 must be equal")

        for i in range(len(train_x)):
            for j in range(len(train_x[i])):
                self.hidden_neurons1[i].inputs.append(Synapse(self.input_neurons[j], 0.5 if train_x[i][j] == 1 else -0.5))

    def test(self, test_x: list[tuple[float]], test_y: list[int]) -> tuple[float, float]:
        correct_ctr = 0
        processed_ctr = 0
        for test_case, expected_result in zip(test_x, test_y):
            processed_ctr += 1
            result = self.predict(test_case)
            if result == expected_result:
                correct_ctr += 1
            else:
                print(f"Expected: {expected_result}, got: {result}")
        return correct_ctr, processed_ctr

    def predict(self, x: tuple[float]) -> int:
        for n, x_val in zip(self.input_neurons, x):
            n.accept_value(1 if x_val > 0 else -1)
        Hamming.feed(self.hidden_neurons1)
        Hamming.clear_values(self.hidden_neurons2)
        Hamming.feed(self.hidden_neurons2)
        while any([n.previous_value != n.out_value for n in self.hidden_neurons2]):
            Hamming.update_previous(self.hidden_neurons2)
            Hamming.feed(self.hidden_neurons2)

        activated_neurons = [n for n in self.hidden_neurons2 if n.out_value > 0]
        return -1 if len(activated_neurons) == 0 else self.hidden_neurons2.index(activated_neurons[0])  # Повертаємо індекс переможця

    @staticmethod
    def feed(neurons: list[Neuron|HammingNeuron]):
        for n in neurons:
            n.feedforward()

    @staticmethod
    def update_previous(neurons: list[HammingNeuron]):
        for n in neurons:
            n.previous_value = n.out_value

    @staticmethod
    def clear_values(neurons: list[HammingNeuron]):
        for n in neurons:
            n.previous_value = 0
            n.out_value = 0
