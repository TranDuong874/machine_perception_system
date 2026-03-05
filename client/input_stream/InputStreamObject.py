from abc import ABC, abstractmethod

class InputStreamObject(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def read_next(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def is_exhausted(self):
        pass