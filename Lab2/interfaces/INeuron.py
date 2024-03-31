from abc import ABC, abstractmethod


class INeuron(ABC):
    @abstractmethod
    def get_value(self) -> float|None:
        pass
