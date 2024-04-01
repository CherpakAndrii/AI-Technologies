from interfaces.INeuron import INeuron


class Synapse:
    weight: float
    source: INeuron

    def __init__(self, source: INeuron, weight: float):
        self.source = source
        self.weight = weight

    def get_signal(self):
        value = self.source.get_value()
        return value * self.weight if value is not None else None

    def error_propagation(self, error, learning_rate):
        self.weight += learning_rate * error * self.source.get_value()
        self.source.error_propagation(error*self.weight, learning_rate)
