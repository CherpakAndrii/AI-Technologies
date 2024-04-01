from itertools import product

from models.InputLayerNeuron import InputLayerNeuron
from models.Neuron import Neuron
from models.Synapse import Synapse
from models.ThresholdActivationFunction import ThresholdActivationFunction


inputs_size = 2
test_cases = product([0, 1], repeat=inputs_size)

if __name__ == '__main__':
    input_neurons = [InputLayerNeuron() for _ in range(inputs_size)]
    input_synapses: list[Synapse] = [Synapse(input_neuron, 1) for input_neuron in input_neurons]

    neuron = Neuron(input_synapses, 1-inputs_size, ThresholdActivationFunction(1))

    for test_case in test_cases:
        for i in range(inputs_size):
            input_neurons[i].accept_value(test_case[i])

        neuron.feedforward()

        value = neuron.get_value()
        print(f"Inputs: {test_case}, output value: {value}")
