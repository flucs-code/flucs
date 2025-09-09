"""Definition of the abstract base for any flucs solver.

Outlines the basic functionality of any solver using
abstract methods.

"""

from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from flucs import FlucsInput
from .flucs_solver_state import FlucsSolverState


T_System = TypeVar("T_System", bound="FlucsSystem")
class FlucsSolver(Generic[T_System], ABC):
    input: FlucsInput
    system: T_System
    state: FlucsSolverState

    @abstractmethod
    def run(self) -> None:
        pass

    def __init__(self, flucs_input: FlucsInput,
                 flucs_system: T_System) -> None:
        self.input = flucs_input
        self.system = flucs_system
        self.state = FlucsSolverState.NOTINITIALISED
