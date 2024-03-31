from interfaces.IActivationFunction import IActivationFunction
from interfaces.INeuron import INeuron
from models.Synapse import Synapse


class Neuron(INeuron):
    weight: float
    inputs: list[Synapse]
    activation_func: IActivationFunction
    out_value: float|None

    def __init__(self, input_synapses: list[Synapse], weight: float, activation_function: IActivationFunction):
        self.inputs = input_synapses
        self.weight = weight
        self.activation_func = activation_function

    def feedforward(self):
        signals_sum = sum([s.get_signal() for s in self.inputs])
        self.out_value = self.activation_func.compute(signals_sum + self.weight)

    def get_value(self) -> float|None:
        value = self.out_value
        self.out_value = None
        return value
