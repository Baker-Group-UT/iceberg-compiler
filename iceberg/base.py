from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from typing import Iterable
import cirq


def mark_check_encoded(method):
    """Mark method to require _encoded=True when called."""
    method._check_encoded = True
    return method


def check_encoded(method):
    """Wrap method to check if self._encoded is True."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_encoded", False):
            raise ValueError("The circuit is not encoded")
        return method(self, *args, **kwargs)
    return wrapper


class AutoCheckEncodedMeta(ABCMeta):
    """
    Metaclass to automatically apply `check_encoded` to methods
    marked with `@mark_check_encoded` in any subclass.
    """

    def __new__(cls, clsname, bases, classdict):
        for name, method in classdict.items():
            if (
                callable(method)
                and not name.startswith("__")
                and not name.endswith("__")
            ):
                for base in bases:
                    if hasattr(base, name) and getattr(
                        getattr(base, name), "_check_encoded", False
                    ):
                        classdict[name] = check_encoded(method)
                        break
        return super().__new__(cls, clsname, bases, classdict)


class CircuitClass(ABC, metaclass=AutoCheckEncodedMeta):
    """
    Abstract base class for any encoded circuit representation.
    """

    circuit: cirq.Circuit
    _encoded: bool = False

    @abstractmethod
    def encode(self, initialize: str | None): ...

    @mark_check_encoded
    @abstractmethod
    def decode(self): ...

    @mark_check_encoded
    @abstractmethod
    def measure(self): ...

    @mark_check_encoded
    @abstractmethod
    def X(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def Z(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def S(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def H(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def CNOT(self, i: int, j: int): ...

    @mark_check_encoded
    @abstractmethod
    def CZ(self, i: int, j: int): ...

    @mark_check_encoded
    @abstractmethod
    def RZZ(self, i: int, j: int, rads: float): ...

    @mark_check_encoded
    @abstractmethod
    def RZ(self, i: int, rads: float): ...