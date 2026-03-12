from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class InputSource(ABC, Generic[T]):
    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read_next(self) -> T | None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def is_exhausted(self) -> bool:
        ...
