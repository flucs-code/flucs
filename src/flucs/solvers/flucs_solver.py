"""Definition of the abstract base for any flucs solver.

Outlines the basic functionality of any solver using
abstract methods.

"""

import enum
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from flucs import FlucsInput


class FlucsSolverState(enum.Enum):
    """Keeps track of what the solver is doing."""
    NOTINITIALISED = enum.auto()
    INITIALISED = enum.auto()
    TIMING = enum.auto()
    RUNNING = enum.auto()
    PAUSED = enum.auto()
    DONE = enum.auto()


T_System = TypeVar("T_System", bound="FlucsSystem")

class FlucsSolver(Generic[T_System], ABC):
    input: FlucsInput
    system: T_System
    state: FlucsSolverState

    @abstractmethod
    def run(self) -> None:
        """Main entry point for the solver."""

    def __init__(self, flucs_input: FlucsInput,
                 flucs_system: T_System) -> None:
        self.input = flucs_input
        self.system = flucs_system
        self.state = FlucsSolverState.NOTINITIALISED
