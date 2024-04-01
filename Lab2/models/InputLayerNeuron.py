from interfaces.INeuron import INeuron


class InputLayerNeuron(INeuron):
    out_value: float|None

    def accept_value(self, input_value: float):
        self.out_value = input_value

    def get_value(self) -> float|None:
        return self.out_value

    def error_propagation(self, error, learning_rate):
        pass
