from interfaces.IActivationFunction import IActivationFunction
from interfaces.INeuron import INeuron
from models.ReLUActivationFunction import ReLUActivationFunction
from models.Synapse import Synapse


class HammingNeuron(INeuron):
    bias: float
    inputs: list[Synapse]
    out_value: float|None
    previous_value: float|None
    activation_func: IActivationFunction

    def __init__(self, input_synapses: list[Synapse], bias: float, activation_func=ReLUActivationFunction()):
        self.inputs = input_synapses
        self.bias = bias
        self.out_value = 0
        self.previous_value = 0
        self.activation_func = activation_func

    def feedforward(self):
        signals_sum = sum([s.get_signal() for s in self.inputs])
        self.out_value = self.activation_func.compute(signals_sum + self.bias)

    def get_value(self) -> float|None:
        return self.previous_value

    def error_propagation(self, error, learning_rate):
        pass
