from interfaces.IActivationFunction import IActivationFunction
from interfaces.INeuron import INeuron
from models.Synapse import Synapse


class Neuron(INeuron):
    bias: float
    inputs: list[Synapse]
    activation_func: IActivationFunction
    out_value: float|None

    def __init__(self, input_synapses: list[Synapse], bias: float, activation_function: IActivationFunction):
        self.inputs = input_synapses
        self.bias = bias
        self.activation_func = activation_function

    def feedforward(self):
        signals_sum = sum([s.get_signal() for s in self.inputs])
        self.out_value = self.activation_func.compute(signals_sum + self.bias)

    def get_value(self) -> float|None:
        # value = self.out_value
        # self.out_value = None
        return self.out_value#value

    def error_propagation(self, error, learning_rate):
        self.bias += learning_rate * error * self.activation_func.derivative(self.out_value)
        for input_synapse in self.inputs:
            input_synapse.error_propagation(error, learning_rate)
