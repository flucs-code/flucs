"""Definition of the abstract base for any flucs solver.

Outlines the basic functionality of any solver using
abstract methods.

"""

import enum
import signal
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from flucs import FlucsInput
from flucs.utilities.messages import flucsprint

if TYPE_CHECKING:
    from flucs.systems import FlucsSystem


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
    interrupted: bool = False

    @abstractmethod
    def run(self) -> None:
        """Main entry point for the solver."""

    def __init__(self, flucs_input: FlucsInput, flucs_system: T_System) -> None:
        self.input = flucs_input
        self.system = flucs_system
        self.state = FlucsSolverState.NOTINITIALISED

        # Handle signals in order to exit cleanly
        def signal_handler(signum, frame):
            flucsprint(f"\nCaught signal {signum}. Exiting cleanly.")
            self.interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGUSR2, signal_handler)
