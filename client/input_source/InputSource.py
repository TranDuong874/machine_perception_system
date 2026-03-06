from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class InputSource(ABC, Generic[T]):
    def __init__(self):
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def read_next(self) -> T | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def is_exhausted(self) -> bool:
        pass
