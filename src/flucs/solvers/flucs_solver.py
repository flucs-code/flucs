"""Definition of the abstract base for any flucs solver.

Outlines the basic functionality of any solver using
abstract methods.

"""

from __future__ import annotations

import enum
import signal
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from flucs import FlucsInput
from flucs.input import InvalidFlucsInputFileError
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

    timestepper: FlucsTimestepper[T_System]
    _supported_timesteppers: ClassVar[dict[str, type[FlucsTimestepper]]]

    @abstractmethod
    def run(self) -> None:
        """Main entry point for the solver."""

    @abstractmethod
    def setup_cuda_definitions(self) -> None:
        """
        Sets up any CUDA definitions (e.g., compile-time constants, flags, etc)
        """

    @abstractmethod
    def register_kernels(self) -> None:
        """Registers kernels (incl. templated kernels) that are to be used."""

    def _create_timestepper(self):
        timestepper_name = self.input["setup.timestepper"]

        # Check choice of timestepper
        if timestepper_name not in self._supported_timesteppers:
            raise InvalidFlucsInputFileError(
                f"{self.input['setup.timestepper']} is not"
                " a supported timestepper. Supported "
                f"choices are {', '.join(self._supported_timesteppers)}."
            )

        self.timestepper = self._supported_timesteppers[timestepper_name](self)

        flucsprint(f"Time integrator: {self.timestepper!s}", source=self)

    def __init__(self, flucs_input: FlucsInput, flucs_system: T_System) -> None:
        self.input = flucs_input
        self.system = flucs_system
        self.state = FlucsSolverState.NOTINITIALISED

        self._create_timestepper()

        # Handle signals in order to exit cleanly
        def signal_handler(signum, frame):
            flucsprint(f"\nCaught signal {signum}. Exiting cleanly.")
            self.interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGUSR2, signal_handler)


class FlucsTimestepper(Generic[T_System], ABC):
    solver: FlucsSolver[T_System]
    system: T_System
    input: FlucsInput

    @abstractmethod
    def setup_cuda_definitions(self):
        """
        Sets up any CUDA definitions (e.g., compile-time constants, flags, etc)
        """

    @abstractmethod
    def perform_timestep(self):
        """
        Performs a complete time step
        """

    @abstractmethod
    def setup(self) -> None:
        """
        Sets up the timestepper for running (e.g., memory allocation)
        """

    @abstractmethod
    def ready(self) -> None:
        """
        This method is called immediately before the solver starts
        execution.

        """

    # Force the user to create a nice string representation
    @abstractmethod
    def __str__(self) -> str:
        pass

    def __init__(self, solver: FlucsSolver[T_System]):
        self.solver = solver
        self.system = solver.system
        self.input = solver.input
